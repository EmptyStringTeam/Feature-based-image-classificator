#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <qfiledialog.h>
#include <fstream>
#include <sys/types.h>
#include <QComboBox>
#include <pwd.h>
#include <limits.h>
#include <math.h>

#include "utils.h"

#include <codebookcontroller.h>
#include <codebookworker.h>
#include <traincontroller.h>
#include <trainworker.h>

using namespace cv;
using namespace std;

///
/// Window creation
///
MainWindow::MainWindow(QWidget *parent) :
    QMainWindow( parent ),
    controller( NULL ),
    ui( new Ui::MainWindow )
{
    ui->setupUi( this );

    ui->featureDetectionMethod->addItem( "SURF" );
    ui->featureDetectionMethod->addItem( "SIFT" );
    ui->featureDetectionMethod->addItem( "KAZE" );

    ui->histogramMethod       ->addItem( "Bag of Words"           );
    ui->histogramMethod       ->addItem( "Fisher Vector" );

    // default paths
    struct  passwd *pw = getpwuid( getuid() );
    homePath = pw->pw_dir;

    logStream.Init( "logfile.log" );

    ui->inputFolder   ->setText( ( homePath + "/Dataset"      ).c_str() );
    ui->inputCodebook ->setText( ( homePath + "/codebook.cbk" ).c_str() );
    ui->outputCodebook->setText( ( homePath + "/codebook.cbk" ).c_str() );

    // Setup status bar
    // status label
    statusLabel = new QLabel( "Ready." );
    statusLabel->setAlignment( Qt::AlignLeft | Qt::AlignVCenter );
    statusLabel->setMinimumSize( statusLabel->sizeHint() );

    // progressbar
    progressBar = new QProgressBar();
    progressBar->setMinimumWidth( this->width()/3 );
    progressBar->setRange( 0, 1 );
    progressBar->setTextVisible( true );
    progressBar->setVisible( false );

    // audio mute button
    audioMuted    = false;
    audioMutedNum = 0;
    audioToggleBtn = new QPushButton();
    QIcon::setThemeName          ( "ubuntu-mono-dark" );
    audioToggleBtn->setIcon      ( QIcon::fromTheme( "audio-volume-high" ) );
    audioToggleBtn->setIconSize  ( QSize( 24, 24 ) );
    audioToggleBtn->setStyleSheet( "QPushButton{border: none;outline: none;}" );

    connect( audioToggleBtn, SIGNAL( clicked() ), this, SLOT( toggleAudio() ) );

    // abort button
    abortBtn = new QToolButton();
    abortBtn->setText   ( "X" );
    abortBtn->setToolTip( "Abort current operation" );
    abortBtn->setVisible( false );

    connect( abortBtn, SIGNAL( clicked() ), this, SLOT( abortOperation() ) );

    statusBar()->setContentsMargins( 3, 0, 3, 0);
    statusBar()->addWidget( statusLabel, 1 );
    statusBar()->addWidget( progressBar    );
    statusBar()->addWidget( abortBtn       );
    statusBar()->addWidget( audioToggleBtn );

    abortRequested = false;
}

MainWindow::~MainWindow()
{
    delete ui;
    if( controller )
    {
        delete controller;
        controller = NULL;
    }
}

///
/// progressbar control signals (for workers)
///
void MainWindow::setupProgressBar( QString inputFolder, int splitPercent )
{
    int file_count = 0;

    vector< string > dir_list;
    if( num_dirs( inputFolder.toStdString().c_str(), dir_list ) < 0 )
        return;

    for( unsigned int dirIx = 0; dirIx < dir_list.size(); dirIx++ )
    {
        char path[256];
        sprintf(path,"%s/%s", inputFolder.toStdString().c_str(), dir_list[ dirIx ].c_str() );

        file_count += floor( num_images( path ) * splitPercent / 100.0f );
    }

    progressBar->setRange ( 0, file_count );
    progressBar->setValue ( 0 );
    progressBar->setFormat( "0%" );

    clock_gettime( CLOCK_REALTIME, &progressBarTimer );
}

void MainWindow::updateProgressbar()
{
    progressBar->setValue( progressBar->value() + 1 );
    float percent = 100.0f * progressBar->value() / progressBar->maximum();

    QString percentStr = QString::number( (int)percent ) + "%";

    if( percent < 2 || percent > 99 )
        progressBar->setFormat( percentStr );
    else
    {
        // Show ETA
        struct timespec actualTime;
        clock_gettime( CLOCK_REALTIME, &actualTime );

        int elapsedTime = actualTime.tv_sec - progressBarTimer.tv_sec;
        int estimatedTime = ( elapsedTime / percent ) * ( 100-percent );
        int etaHours = 0;
        int etaMins  = estimatedTime / 60;

        if( etaMins > 60 )
        {
            etaHours = etaMins / 60;
            etaMins = etaMins % 60;
        }

        int etaSecs  = estimatedTime % 60;

        etaSecs = ceil( etaSecs / 10.0f ) * 10;

        if( etaSecs == 60 )
        {
            etaSecs = 0;
            etaMins++;
        }

        QString etaString;
        if( etaHours>0 )
            etaString = QString::number( etaHours ) + "h, " + ( etaMins < 10 ? "0" : "" ) + QString::number( etaMins ) + "m";
        else if( etaMins>0 || etaSecs>10 )
            etaString = QString::number( etaMins  ) + "m, " + ( etaSecs < 10 ? "0" : "" ) + QString::number( etaSecs ) + "s";
        else
            etaString = "<10s";

        progressBar->setFormat( percentStr + " - ETA: " + etaString );
    }
}

///
/// various UI events
///
void MainWindow::on_inputCodebookBtn_clicked()
{
    QString fileName = QFileDialog::getOpenFileName( this, tr( "Select input codebook" ), homePath.c_str(), tr( "Codebook files (*.cbk)" ) );

    if( !fileName.isEmpty() )
        ui->inputCodebook->setText( fileName );
}

void MainWindow::on_inputFolderBtn_clicked()
{
    QString folder =  QFileDialog::getExistingDirectory( this, tr( "Select input folder" ), homePath.c_str(), QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks );

    if( !folder.isEmpty() )
        ui->inputFolder->setText( folder );
}

void MainWindow::on_outputCodebookBtn_clicked()
{
    QString fileName = QFileDialog::getSaveFileName (this, tr("Select output codebook"), homePath.c_str(), tr( "Codebook files (*.cbk)" ) );

    if( !fileName.isEmpty() )
        ui->outputCodebook->setText( fileName );
}

void MainWindow::toggleAudio()
{
    audioMuted = !audioMuted;
    audioMutedNum++;

    audioToggleBtn->setIcon      ( QIcon::fromTheme( audioMuted ? "audio-volume-muted" : ( audioMutedNum>3 ? "audio-volume-low" : "audio-volume-high" )  ) );
    audioToggleBtn->setIconSize  ( QSize( 24, 24 ) );
    audioToggleBtn->setStyleSheet( "QPushButton{border: none;outline: none;}" );
}

void MainWindow::on_splitSlider_valueChanged()
{
    int sliderValue = ui->splitSlider->value();
    char splitText[256];

    sprintf( splitText, "Training %d%%, testing %d%%", sliderValue, 100-sliderValue );
    ui->splitLabel->setText( splitText );
}

void MainWindow::on_featureDetectionMethod_currentIndexChanged(int)
{
    ui->minHessian->setEnabled( ui->featureDetectionMethod->currentText() == "SURF" );
    ui->label_2   ->setEnabled( ui->featureDetectionMethod->currentText() == "SURF" );
}

void MainWindow::on_inputCodebook_textChanged()
{
    updateCodebookLabel();
}

void MainWindow::on_outputCodebook_textChanged()
{
    ui->inputCodebook->setText( ui->outputCodebook->text() );
}

///
/// codebook creation signals
///
void MainWindow::codebook_imgReadingDone()
{
    statusLabel->setText ( "Codebook creation: clustering.." );
    progressBar->setRange( 0, 0 );
    abortBtn   ->setEnabled( false );
}

void MainWindow::codebook_clusteringDone()
{
    playSuccess();
    updateCodebookLabel();
}

///
/// training and testing signals
///
void MainWindow::training_imgReadingDone()
{
    statusLabel->setText ( "SVM training.." );
    progressBar->setRange( 0, 0 );
    abortBtn   ->setEnabled( false );
}

void MainWindow::training_trainingDone()
{
    statusLabel->setText( "Testing images.." );
    abortBtn   ->setEnabled( true );
}

void MainWindow::training_testingDone( float accuracy )
{
    char status[256];
    sprintf( status, "Testing finished! Accuracy: %d%%", (int)accuracy );

    Mat image;
    image = imread( "confusion_matrix.png" );
    imshow( "Confusion Matrix", image ); // show it to the user

    statusLabel->setText( status );
    playSuccess();
}


///
/// common signals
///
void MainWindow::processingDone()
{
    enableControls ( true );
    logStream.flush();

    ui->codebookBtn->setText( "Create codebook" );
    ui->trainBtn   ->setText( "SVM Train" );

    statusLabel    ->setText( "Ready." );
    progressBar    ->setVisible( false );
    abortBtn       ->setVisible( false );

    updateCodebookLabel();

    delete controller;
}

void MainWindow::throwError(const QString &error)
{
    processingDone();

    if( abortRequested )
        statusLabel->setText( error );
    else
    {
        statusLabel->setText( "An error has occurred! " + error );
        playError();
    }
}

///
/// abort current operation
///
void MainWindow::abortOperation()
{
    abortRequested = true;
    abortBtn->setEnabled( false );
    /*if( controller )
    {
        delete controller;
        controller = NULL;

        processingDone();
    }*/
}

void MainWindow::checkAbort(bool *abortRequested)
{
    *abortRequested = this->abortRequested;
}

///
/// workers creation
///
void MainWindow::on_codebookBtn_clicked()
{
    QString method      = ui->featureDetectionMethod->currentText();
    QString inputFolder = ui->inputFolder   ->text();
    QString outputFile  = ui->outputCodebook->text();
    int     splitPerc   = ui->splitSlider   ->value();
    int     minHessian  = ui->minHessian    ->value();
    int     clusterSize = ui->clusterSize   ->value();

    // tilde to homePath
    inputFolder = toLocalPath( inputFolder );
    outputFile  = toLocalPath( outputFile  );

    controller = new CodebookController( this, method, inputFolder, outputFile, splitPerc, minHessian, clusterSize );

    enableControls( false );
    ui->codebookBtn->setText( "Processing..." );
    statusLabel    ->setText( "Codebook creation: processing images.." );

    progressBar->setValue  ( 0 );
    progressBar->setVisible( true );
    abortBtn   ->setVisible( true );
    abortBtn   ->setEnabled( true );

    abortRequested = false;
}

void MainWindow::on_trainBtn_clicked()
{
    QString inputFile       = ui->inputCodebook  ->text();
    QString histogramMethod = ui->histogramMethod->currentText();

    // tilde to homePath
    inputFile = toLocalPath( inputFile );

    controller = new TrainController( this, inputFile, histogramMethod );

    enableControls( false );
    ui->trainBtn->setText( "Processing..." );
    statusLabel ->setText( "Creating images histograms.." );

    progressBar->setValue  ( 0 );
    progressBar->setVisible( true );
    abortBtn   ->setVisible( true );
    abortBtn   ->setEnabled( true );

    abortRequested = false;
}

///
/// toggle controls enabled status
///
void MainWindow::enableControls( bool enable )
{
    ui->codebookBtn           ->setEnabled( enable );
    ui->trainBtn              ->setEnabled( enable );
    ui->inputFolder           ->setEnabled( enable );
    ui->inputFolderBtn        ->setEnabled( enable );
    ui->inputCodebook         ->setEnabled( enable );
    ui->inputCodebookBtn      ->setEnabled( enable );
    ui->outputCodebook        ->setEnabled( enable );
    ui->outputCodebookBtn     ->setEnabled( enable );
    ui->splitSlider           ->setEnabled( enable );
    ui->minHessian            ->setEnabled( enable && ui->featureDetectionMethod->currentText() == "SURF" );
    ui->clusterSize           ->setEnabled( enable );
    ui->histogramMethod       ->setEnabled( enable );
    ui->featureDetectionMethod->setEnabled( enable );
}

///
/// audio
///
void MainWindow::playSuccess()
{
    if( !audioMuted )
    {
        char command[128];
        sprintf( command, "aplay -qN  success.wav");
        if( audioMutedNum > 3 )
            command[10] = '.';
        system( command );
    }
}

void MainWindow::playError()
{
    if( !audioMuted )
    {
        char command[128];
        sprintf( command, "aplay -qN  error.wav");
        if( audioMutedNum > 3 )
            command[10] = '.';
        system( command );
    }
}

///
/// selected codebook metadata preview
///
void MainWindow::updateCodebookLabel()
{
    fstream pathStream;
    stringstream labelStream;
    QString inputCodebook = ui->inputCodebook->text();

    // tilde to homePath
    inputCodebook = toLocalPath( inputCodebook );

    pathStream.open( inputCodebook.toStdString().c_str(), ios::in );
    if( pathStream.is_open() )
    {
        if( readCodebookMetadata( pathStream ) )
        {
            labelStream << "Dataset: " << datasetName << " (" << splitPercent
                        << "% training, " << 100-splitPercent << "% testing)\n"
                        << "Method: " << method;
            if( method=="SURF" )
                labelStream << ", Min Hessian: " << minHessian;
            labelStream << ", Codebook clusters: " << clusterSize << "\n";

            lastModified = fileLastModification( inputCodebook.toStdString().c_str() );

            time_t curr_time;
            time ( &curr_time );

            if( curr_time - lastModified < 60*60 )
                labelStream << "Created " << (curr_time - lastModified) / 60 << " minutes ago";
            else
                labelStream << "Created on " << ctime( &lastModified );
        }
        else
        {
            labelStream << "Invalid codebook file!";
        }
    }

    ui->inputCodebookLabel->setText( labelStream.str().c_str() );
    pathStream.close();
}

bool MainWindow::readCodebookMetadata( fstream& fin )
{
    string row;
    size_t pos, end;

    // first row contains all metadata: method, minHessian, dataset folder, split percent, cluster size.
    getline( fin, row );

    // extract feature-detection method used
    if( row.find( "SIFT" ) != string::npos )
    {
        method = "SIFT";
    }
    else if( row.find( "SURF" ) != string::npos )
    {
        method = "SURF";
    }
    else if( row.find( "KAZE" ) != string::npos )
    {
        method = "KAZE";
    }
    else
    {
        return false;
    }

    // extract dataset folder
    pos = row.find( "dataset: '" );
    pos += 10;
    end = row.find( "'", pos );
    if( pos==string::npos || end==string::npos )
        return false;

    datasetFolder = row.substr( pos, end-pos );
    pos = datasetFolder.find_last_of('/', pos) + 1;
    datasetName = datasetFolder.substr( pos );


    // extract dataset split percent
    end = row.rfind( '%' );
    pos = row.rfind( ' ', end );
    if( pos==string::npos || end==string::npos )
        return false;

    splitPercent = atoi( row.substr( pos, end ).c_str() );
    if( splitPercent<1 || splitPercent > 99 )
        return false;

    // extract minHessian
    if( method == "SURF" )
    {
        pos = row.find( "minHessian: " );
        end = row.find( ",", pos );
        if( pos==string::npos || end==string::npos )
            return false;

        minHessian = atoi( row.substr( pos+12, end ).c_str() );
        if( minHessian<100 || minHessian > 10000 )
            return false;
    }
    else
    {
        minHessian = 0;
    }

    // extract clusterSize
    pos = row.find( "clusterSize: " );
    end = row.find( ",", pos );
    if( pos==string::npos || end==string::npos )
        return false;

    clusterSize = atoi( row.substr( pos+13, end ).c_str() );
    if( clusterSize<100 || clusterSize > 10000 )
        return false;

    // extract descriptorSize
    pos = row.find( "descriptorsSize: " );
    end = row.find( ",", pos );
    if( pos==string::npos || end==string::npos )
        return false;

    descriptorSize = atoi( row.substr( pos+17, end ).c_str() );
    if( descriptorSize<64 || descriptorSize > 128 )
        return false;

    return true;
}
