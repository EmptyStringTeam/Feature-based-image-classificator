#include "trainworker.h"
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <iomanip>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include "utils.h"

extern "C"
{
    #include <vl/generic.h>
    #include <vl/gmm.h>
    #include <vl/fisher.h>
}

using namespace std;
using namespace cv;

TrainWorker::TrainWorker()
{
}

void TrainWorker::doWork( QString inputFile, QString histogramMethod )
{
    this->histogramMethod = histogramMethod.toStdString();
    datasetSize = 0;

    struct timespec start, readingTime, trainingTime, testingTime;
    vector< string > dir_list;
    fstream fin;

    fin.open( inputFile.toStdString().c_str(), ios::in );

    if( !fin.is_open() )
    {
        logStream << "Unable to open codebook file '" << inputFile.toStdString() << "'\n";
        throwError( "Unable to open codebook file." );
        return;
    }

    if( !readCodebookMetadata( fin ) )
        return;

    // replace tilde with homePath
    datasetFolder = toLocalPath( datasetFolder );

    Mat codebook( clusterSize, descriptorSize, CV_32FC1 );
    readCodebook( fin, codebook );

    fin.close();

    logStream << "\nTESTING -- using method: " << method;
    if( method == "SURF" )
        logStream << ", minHessian: " << minHessian;
    logStream << ", clusterSize: " << clusterSize
              << ", histograms creation: " << this->histogramMethod << "\n";

    Feature2D *detector  = NULL;
    Feature2D *extractor = NULL;

    if( method == "SIFT" )
    {
        detector  = new SiftFeatureDetector    ();
        extractor = new SiftDescriptorExtractor();
    }
    else if( method == "SURF" )
    {
        detector  = new SurfFeatureDetector    ( minHessian );
        extractor = new SurfDescriptorExtractor();
    }
    else if( method == "KAZE" )
    {
        detector  = new KAZE();
        extractor = new KAZE();
    }

    if( detector == NULL || extractor == NULL )
    {
        logStream << "Initialization error, aborting.\n";
        throwError( ( method + " initialization error." ).c_str()  );
        return;
    }

    // setup GMM, if using fisher vector
    gmm = NULL;
    if( histogramMethod == "Fisher Vector" )
        setupGMM( codebook );

    if( num_dirs( datasetFolder.c_str(), dir_list ) < 0 )
    {
        logStream << "Incorrect input folder, aborting.\n";
        throwError( "Incorrect input folder" );
        return;
    }

    clock_gettime( CLOCK_REALTIME, &start );

    int totalFilesCount = 0;

    vector< int > labels;
    vector< vector< float > > histograms;

    emit setupProgressBar( QString( datasetFolder.c_str() ), splitPercent );

    // Create file list using splitPercent files of each directory
    vector < pair < int, string > > fileList;
    for( unsigned int dirIx = 0; dirIx<dir_list.size(); dirIx++ )
    {
        categories.push_back( dir_list[ dirIx ] );

        char path[256];
        sprintf( path,"%s/%s", datasetFolder.c_str(), dir_list[ dirIx ].c_str() );

        vector< string > pathFileList;

        int dirFileCount   = num_images( path, pathFileList );
        int trainFileCount = floor( dirFileCount * splitPercent / 100.0f );

        for( int fileIx = 0; fileIx < trainFileCount; fileIx++ )
        {
            sprintf( path,"%s/%s/%s", datasetFolder.c_str(), dir_list[ dirIx ].c_str(), pathFileList[ fileIx ].c_str() );
            fileList.push_back( pair<int,string>( dirIx, path ) );
        }
    }

    // Start extracting keypoints from each image
    #pragma omp parallel for shared(labels,histograms,detector,extractor)
    for( unsigned int fileIx = 0; fileIx<fileList.size(); fileIx++ )
    {
        bool abortRequested = false;
        emit checkAbort( &abortRequested );

        if( abortRequested )
        {
            fileIx += 10000; // to force exit
            continue;
        }

        Mat img = imread( fileList[ fileIx ].second.c_str(), 1 );

        cout << toGlobalPath( fileList[ fileIx ].second ) << endl << flush; // output current filename (cout only, not to logFile)

        vector< KeyPoint > keypoints;
        detector->detect( img, keypoints );

        Mat descriptors;
        extractor->compute( img, keypoints, descriptors );

        vector< float > histogram;
        histogram = createHistogram( codebook, descriptors );

        //
        #pragma omp critical
        {
            labels.push_back( fileList[ fileIx ].first );
            histograms.push_back( histogram );

            totalFilesCount++;
        }

        emit updateProgressbar();
    }

    bool abortRequested = false;
    emit checkAbort( &abortRequested );

    if( abortRequested )
    {
        logStream << "Operation aborted by user.\n\n";
        emit throwError( "Operation aborted" );
        return;
    }

    emit imgReadingDone();
    clock_gettime( CLOCK_REALTIME, &readingTime );

    // prepare SVM training data
    Mat histogramsMat( histograms.size(), histograms.at(0).size(), CV_32FC1 );
    for( int i=0; i<histogramsMat.rows; i++ )
        for( int j=0; j<histogramsMat.cols; j++ )
            histogramsMat.at<float>(i, j) = histograms.at(i).at(j);

    // create and train SVM classifiers
    vector< CvSVM* > classifiers;
    for( unsigned int cat = 0; cat < categories.size(); cat++ )
    {
        CvSVMParams params;

        params.C           = 1;
        params.gamma       = 0.001;
        params.kernel_type = CvSVM::RBF;
        params.svm_type    = CvSVM::C_SVC;

        CvSVM* svm = new CvSVM();
        Mat labelsMat( labels.size(), 1, CV_32FC1 );

        for( int i=0; i<labelsMat.rows; i++ )
        {
            if( labels[i] == (int)cat )
                labelsMat.at<float>(i) = 1.0f;
            else
                labelsMat.at<float>(i) = -1.0f;
        }

        Mat varIdx, sampleIdx;
        svm->train( histogramsMat, labelsMat, varIdx, sampleIdx, params );

        classifiers.push_back( svm );
    }

    emit trainingDone();
    clock_gettime( CLOCK_REALTIME, &trainingTime );
    datasetSize = totalFilesCount;

    // Testing
    emit setupProgressBar( QString( datasetFolder.c_str() ), 100 - splitPercent );

    // Create file list using (100-splitPercent) files of each directory
    fileList.clear();
    for( unsigned int dirIx = 0; dirIx<dir_list.size(); dirIx++ )
    {
        char path[256];
        sprintf( path,"%s/%s", datasetFolder.c_str(), dir_list[ dirIx ].c_str() );

        vector< string > pathFileList;

        int dirFileCount   = num_images( path, pathFileList );
        int trainFileCount = floor( dirFileCount * splitPercent / 100.0f );

        for( int fileIx = trainFileCount; fileIx < dirFileCount; fileIx++ )
        {
            sprintf( path,"%s/%s/%s", datasetFolder.c_str(), dir_list[ dirIx ].c_str(), pathFileList[ fileIx ].c_str() );
            fileList.push_back( pair<int,string>( dirIx, path ) );
        }
    }

    // confusion matrix
    vector< vector< int > > confusion;
    confusion.resize( categories.size() );
    for( unsigned int i = 0; i<confusion.size(); i++ )
        confusion[i].resize( categories.size() );

    totalFilesCount = 0;
    // Test each image
    #pragma omp parallel for shared(detector,extractor,classifiers,confusion)
    for (unsigned int fileIx = 0; fileIx<fileList.size(); fileIx++)
    {
        bool abortRequested = false;
        emit checkAbort( &abortRequested );

        if( abortRequested )
        {
            fileIx += 10000; // to force exit
            continue;
        }

        Mat img = imread( fileList[ fileIx ].second.c_str(), 1 );

        vector< KeyPoint > keypoints;
        detector->detect(img, keypoints);

        Mat descriptors;
        extractor->compute( img, keypoints, descriptors );

        vector< float > histogram;
        histogram = createHistogram( codebook, descriptors );

        Mat histogramMat(1, histogram.size(), CV_32FC1);
        for( int i=0; i<histogramMat.cols; i++ )
            histogramMat.at<float>(i) = histogram[i];

        int response = -1;
        float minDist = FLT_MAX;

        #pragma omp critical
        for( unsigned int i = 0; i < classifiers.size(); i++ )
        {
            float dist = classifiers[i]->predict( histogramMat, true );
            if( dist<minDist )
            {
                minDist = dist;
                response = i;
            }
        }

        bool correct = (response == fileList[ fileIx ].first);
        logStream << toGlobalPath( fileList[ fileIx ].second ) << " belongs to category: " << categories[ response ];
        logStream << ( correct? " CORRECT!\n" : " WRONG!\n" );

        #pragma omp critical
        {
            confusion[fileList[ fileIx ].first][response]++;
            totalFilesCount++;
        }

        emit updateProgressbar();
    }

    abortRequested = false;
    emit checkAbort( &abortRequested );

    if( abortRequested )
    {
        logStream << "Operation aborted by user.\n\n";
        emit throwError( "Operation aborted" );
        return;
    }


    clock_gettime( CLOCK_REALTIME, &testingTime );
    datasetSize += totalFilesCount;

    // print times
    int readingMins  = ( readingTime .tv_sec - start.tv_sec ) / 60;
    int readingSecs  = ( readingTime .tv_sec - start.tv_sec ) % 60;
    int trainingMins = ( trainingTime.tv_sec - readingTime.tv_sec ) / 60;
    int trainingSecs = ( trainingTime.tv_sec - readingTime.tv_sec ) % 60;
    int testingMins  = ( testingTime .tv_sec - trainingTime.tv_sec ) / 60;
    int testingSecs  = ( testingTime .tv_sec - trainingTime.tv_sec ) % 60;

    // log accuracy, confusion matrix, precision and recall
    float acc = accuracy( confusion );

    logStream << "\n";
    logStream << "Image reading time: " << readingMins  << "m, " << readingSecs  << "s.\n";
    logStream << "Training time:      " << trainingMins << "m, " << trainingSecs << "s.\n";
    logStream << "Testing time:       " << testingMins  << "m, " << testingSecs  << "s.\n";
    logStream << "\n";
    logStream << "Confusion Matrix:\n";
    for( unsigned int i = 0; i< confusion.size(); i++ )
    {
        logStream << categories[ i ];
        for( unsigned int j = 12; j>categories[i].length(); j-- ) // nice formatting
            logStream << " ";
        for( unsigned int j = 0; j<confusion.size(); j++ )
            logStream << "[" << setw(3) << confusion[i][j] << "]";
        logStream << "\n";
    }
    logStream << "Accuracy:   " << acc << "%\n";

    logStream << "Recall:\n";
    for( unsigned int i = 0; i<confusion.size(); i++ )
    {
        float recallValue = recall( confusion[i], i );
        logStream << "            " << categories[i] << ": " << recallValue << "%\n";
    }

    logStream << "Precision:\n";
    for( unsigned int i = 0; i<confusion.size(); i++ )
    {
        vector< int > tmp;
        for( unsigned int j = 0; j<confusion.size(); j++ )
            tmp.push_back( confusion[j][i] );

        float precisionValue = precision( tmp, i );
        logStream << "            " << categories[i] << ": " << precisionValue << "%\n";
    }

    // create and save 'confusion_matrix.png'
    createConfusionMatrixImg( confusion );

    emit processingDone();
    emit testingDone   ( acc );

    /// cleanup
    if( gmm )
        vl_gmm_delete( gmm );
    codebook.deallocate();
    for( unsigned int i = 0; i < histograms.size(); i++ )
        histograms[ i ].clear();
    histograms.clear();
    histogramsMat.deallocate();

    for( unsigned int i = 0; i < confusion.size(); i++ )
        confusion[ i ].clear();
    confusion.clear();

    delete detector;
    delete extractor;
    dir_list.clear();
    fileList.clear();

    for( unsigned int i = 0; i < classifiers.size(); i++ )
        delete classifiers[ i ];
    classifiers.clear();
}

vector< float > TrainWorker::createHistogram( const Mat &codebook, const Mat &descriptors )
{
    if( histogramMethod == "Bag of Words" )
        return createBOWHistogram   ( codebook, descriptors );
    else
        return createFisherHistogram( descriptors, codebook.cols );
}

vector< float > TrainWorker::createBOWHistogram( const Mat &codebook, const Mat &descriptors )
{
    vector< float > histogram;
    float maxRec = 0;

    histogram.resize( codebook.rows );

    for( int dix = 0; dix<descriptors.rows; dix++ )
    {
        long double minDist       = LDBL_MAX;
        int index = -1;

        for( int vix = 0; vix < codebook.rows; vix++ )
        {
            long double distance = 0;
            for( int i = 0; i<codebook.cols; i++ )
            {
                long double t = descriptors.at<float>(dix,i) - codebook.at<float>(vix,i);

                distance += t*t;
            }

            distance = sqrt(distance);

            if( distance < minDist )
            {
                minDist = distance;
                index   = vix;
            }
        }

        if( index > -1 )
            maxRec = max( maxRec, ++histogram[index] );
        else
            cerr << "OVERFLOW";
    }

    // Normalize between 0 and maxValue
    //for( unsigned int i = 0; i<histogram.size(); i++ )
    //    histogram[i] = histogram[i] / maxRec;

    // Normalize using logistic regression function ( 1/1+e^-x )
    //for( unsigned int i = 0; i<histogram.size(); i++)
    //  histogram[i] = 1.0f / ( 1+exp( -histogram[i] ) );

    return histogram;
}

vector< float > TrainWorker::createFisherHistogram( const Mat &descriptors, int dimension )
{
    vector< float > histogram;  

    float *means       = (float*)vl_gmm_get_means      ( gmm );
    float *covariances = (float*)vl_gmm_get_covariances( gmm );
    float *priors      = (float*)vl_gmm_get_priors     ( gmm );

    // get the soft assignments of the data points to each cluster
    //float* posteriors = (float*)vl_gmm_get_posteriors( gmm );

    // encoding
    float* enc;

    // allocate space for the encoding
    enc = (float*)vl_malloc( sizeof( float ) * 2 * dimension * clusterSize );

    // run fisher encoding
    vl_fisher_encode( enc, VL_TYPE_FLOAT,
                      means, dimension, clusterSize,
                      covariances,
                      priors,
                      descriptors.data, descriptors.rows,
                      VL_FISHER_FLAG_IMPROVED );

    for( int i = 0; i < 2 * dimension * clusterSize; i++ )
        histogram.push_back( enc[i] );

    vl_free( enc );

    return histogram;
}

void TrainWorker::setupGMM( const Mat &codebook )
{
    // create a new instance of a GMM object for float data
    gmm = vl_gmm_new( VL_TYPE_FLOAT, codebook.cols, clusterSize );

    // set the maximum number of EM iterations to 100
    vl_gmm_set_max_num_iterations( gmm, 1000 );

    // set the initialization to random selection
    vl_gmm_set_initialization( gmm, VlGMMKMeans );

    // cluster the data, i.e. learn the GMM
    vl_gmm_cluster( gmm, codebook.data, codebook.rows );
}

float TrainWorker::accuracy( const vector< vector< int > > &confusion )
{
    float diagonal = 0;
    float total    = 0;

    for( unsigned int i = 0; i<confusion.size(); i++ )
        for( unsigned int j = 0; j<confusion.size(); j++ )
        {
            total += confusion[i][j];
            if( i == j)
                diagonal += confusion[i][j];
        }

    return ( diagonal/total ) * 100;
}

float TrainWorker::precision( const vector <int> col, int index )
{
    float total = 0;
    for( unsigned int i = 0; i<col.size(); i++ )
        total += col[i];

    float val = col[index];

    if( total == 0 )
        return 0;

    return ( val/total ) * 100;
}

float TrainWorker::recall( const vector <int> row, int index )
{
    float total = 0;
    for( unsigned int i = 0; i<row.size(); i++ )
        total += row[i];

    float val = row[index];

    if( total == 0 )
        return 0;

    return ( val/total ) * 100;
}

void TrainWorker::createConfusionMatrixImg( const vector< vector< int > > &confusion )
{
    // you don't want to know what happens here.
    const int imgSize   = 768;
    const int border    = 128;
    const int baseColor = 200;

    Mat image = Mat::zeros( imgSize, imgSize, CV_8UC3 );

    // no, really
    for( int i = 0; i < image.rows; i++ )
        for( int j = 0; j < image.cols; j++ )
            image.at<Vec3b>(i,j) = Vec3b(255,255,255);

    int   drawArea = imgSize  - 2*border;
    float cellSize = drawArea / confusion.size();
          drawArea = (int)cellSize * confusion.size();

    int maxValue = 0;
    for( unsigned int i = 0; i < confusion.size(); i++ )
        for( unsigned int j = 0; j < confusion.size(); j++ )
            maxValue = max( maxValue, confusion[i][j] );

    for( int i = border; i < border+drawArea; i++ )
        for( int j = border; j < border+drawArea; j++ )
        {
            int posX = min( (int)confusion.size()-1, (int)floor( (i - border) / cellSize ) );
            int posY = min( (int)confusion.size()-1, (int)floor( (j - border) / cellSize ) );

            int ton = round( ( (float)confusion[posX][posY] / maxValue ) * baseColor );
            Vec3b color( baseColor-ton, baseColor, baseColor-ton );
            image.at<Vec3b>(i,j) = color;
        }

    for( unsigned int i = 0; i < confusion.size(); i++ )
    {
        string labelstring = categories[i];
        std::transform( labelstring.begin(), labelstring.end(),     labelstring.begin(), ::tolower );
        std::transform( labelstring.begin(), labelstring.begin()+1, labelstring.begin(), ::toupper );

        char label[5];
        sprintf( label, "%s", labelstring.c_str() );

        Size  textSize = getTextSize( label, FONT_HERSHEY_SIMPLEX, 0.5, 1, 0);

        // col names
        Point textPoint( border + (i+0.5)*cellSize - textSize.width / 2, border - 10 - textSize.height );
        putText( image, label, textPoint, FONT_HERSHEY_SIMPLEX, 0.5, cvScalar(0,0,0), 1, CV_AA );

        // row names
        textPoint = Point( border - 10 - textSize.width, border + (i+0.5)*cellSize + textSize.height / 2 );
        putText( image, label, textPoint, FONT_HERSHEY_SIMPLEX, 0.5, cvScalar(0,0,0), 1, CV_AA );

        for( unsigned int j = 0; j < confusion.size(); j++ )
        {
            line( image, Point( border           , border+i*cellSize ), Point( border+drawArea,   border+i*cellSize ), Scalar(0,0,0) );
            line( image, Point( border+j*cellSize, border            ), Point( border+j*cellSize, border+drawArea   ), Scalar(0,0,0) );

            char label[5];
            sprintf( label, "%d", confusion[i][j] );

            Size  textSize = getTextSize( label, FONT_HERSHEY_SIMPLEX, 1.0, 1, 0);

            Point textPoint( border + (j+0.5)*cellSize - textSize.width / 2, border + (i+0.5)*cellSize + textSize.height / 2 );
            putText( image, label, textPoint, FONT_HERSHEY_SIMPLEX, 1.0, cvScalar(0,0,0), 1, CV_AA );
        }
    }

    line( image, Point( border                          , border+confusion.size()*cellSize ), Point( border+confusion.size()*cellSize, border+confusion.size()*cellSize ), Scalar(0,0,0) );
    line( image, Point( border+confusion.size()*cellSize, border                           ), Point( border+confusion.size()*cellSize, border+confusion.size()*cellSize ), Scalar(0,0,0) );

    char text[128];
    sprintf( text, "Confusion Matrix" );
    int lineHeight      = getTextSize( text, FONT_HERSHEY_SIMPLEX, 1.0, 1, 0).height + 5;
    int smallLineHeight = getTextSize( text, FONT_HERSHEY_SIMPLEX, 0.8, 1, 0).height + 5;

    putText( image, text, Point(30,30), FONT_HERSHEY_SIMPLEX, 1.0, cvScalar(0,0,0), 1, CV_AA );

    // dataset
    string dataset = datasetFolder.substr( datasetFolder.rfind("/")+1 );
    sprintf( text, "Dataset: \"%s\" - %d images (%d%% training, %d%% testing)", dataset.c_str(), datasetSize, splitPercent, 100-splitPercent );
    putText( image, text, Point( 30, imgSize-3*smallLineHeight ), FONT_HERSHEY_SIMPLEX, 0.5, cvScalar(0,0,0), 1, CV_AA);

    // method, splitPercent, minHessian
    if( method=="SURF" )
        sprintf( text, "Feature-detection method: %s (minHessian: %d), Codebook clusters: %d", method.c_str(), minHessian, clusterSize );
    else
        sprintf( text, "Feature-detection method: %s, Codebook clusters: %d", method.c_str(), clusterSize );

    putText( image, text, Point( 30, imgSize-2*smallLineHeight ), FONT_HERSHEY_SIMPLEX, 0.5, cvScalar(0,0,0), 1, CV_AA );

    sprintf( text, "Images histograms creation method: %s", histogramMethod.c_str() );
    putText( image, text, Point( 30, imgSize-1*smallLineHeight ), FONT_HERSHEY_SIMPLEX, 0.5, cvScalar(0,0,0), 1, CV_AA);

    // Accuracy, Precision, Recall
    sprintf( text, "Accuracy: %d%%", (int)accuracy( confusion ) );

    putText( image, text,         Point(512,30             ), FONT_HERSHEY_SIMPLEX, 1.0, cvScalar(  0,0,  0), 1, CV_AA );
    putText( image, "Recall:",    Point(512,30+1*lineHeight), FONT_HERSHEY_SIMPLEX, 0.8, cvScalar(128,0,  0), 1, CV_AA );
    putText( image, "Precision:", Point(512,30+2*lineHeight), FONT_HERSHEY_SIMPLEX, 0.8, cvScalar(  0,0,128), 1, CV_AA );

    circle( image, Point(700,25+1*lineHeight), 1, cvScalar(128,0,  0), 10 );
    circle( image, Point(700,25+2*lineHeight), 1, cvScalar(  0,0,128), 10 );

    for( unsigned int i = 0; i<confusion.size(); i++ )
    {
        sprintf( text, "%d%%", (int)recall( confusion[i], i ) );

        Size  textSize = getTextSize( text, FONT_HERSHEY_SIMPLEX, 1.0, 1, 0);
        Point textPoint( border + confusion.size()*cellSize + 10, border + (i+0.5)*cellSize + textSize.height / 2 );
        putText( image, text, textPoint, FONT_HERSHEY_SIMPLEX, 1.0, cvScalar(128,0,0), 1, CV_AA );
    }

    for( unsigned int i = 0; i<confusion.size(); i++ )
    {
        vector< int > tmp;
        for( unsigned int j = 0; j<confusion.size(); j++ )
            tmp.push_back( confusion[j][i] );

        sprintf( text, "%d%%", (int)precision( tmp, i ) );

        Size  textSize = getTextSize( text, FONT_HERSHEY_SIMPLEX, 1.0, 1, 0);
        Point textPoint( border + (i+0.5)*cellSize - textSize.width / 2, border + confusion.size()*cellSize + 10 + textSize.height );
        putText( image, text, textPoint, FONT_HERSHEY_SIMPLEX, 1.0, cvScalar(0,0,128), 1, CV_AA );
    }

    imwrite( "confusion_matrix.png", image);
}

bool TrainWorker::readCodebookMetadata( fstream& fin )
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
        logStream << "Unknown feature detection method in input file.\n";
        throwError( "Unknown feature detection method in input file." );
        return false;
    }

    // extract dataset folder
    pos = row.find( "dataset: '" );
    pos += 10;
    end = row.find( "'", pos );
    if( pos==string::npos || end==string::npos )
    {
        logStream << "No dataset folder specified in input file.\n";
        throwError( "Input file error." );
        return false;
    }
    datasetFolder = row.substr( pos, end-pos );

    // extract dataset split percent
    end = row.rfind( '%' );
    pos = row.rfind( ' ', end );
    if( pos==string::npos || end==string::npos )
    {
        logStream << "No dataset split percent specified in input file.\n";
        throwError( "Input file error." );
        return false;
    }
    splitPercent = atoi( row.substr( pos, end ).c_str() );
    if( splitPercent < 1 || splitPercent > 99 )
    {
        logStream << "Invalid training percent value in input file.\n";
        throwError( "Invalid training percent value in input file." );
        return false;
    }

    // extract minHessian
    if( method == "SURF" )
    {
        pos = row.find( "minHessian: " );
        end = row.find( ",", pos );
        if( pos==string::npos || end==string::npos )
        {
            logStream << "No minHessian specified in input file.\n";
            throwError( "Input file error." );
            return false;
        }
        minHessian = atoi( row.substr( pos+12, end ).c_str() );
        if( minHessian < 100 || minHessian > 10000 )
        {
            logStream << "Invalid minHessian value in input file.\n";
            throwError( "Invalid minHessian value in input file." );
            return false;
        }
    }

    // extract clusterSize
    pos = row.find( "clusterSize: " );
    end = row.find( ",", pos );
    if( pos==string::npos || end==string::npos )
    {
        logStream << "No minHessian specified in input file.\n";
        throwError( "Input file error." );
        return false;
    }
    clusterSize = atoi( row.substr( pos+13, end ).c_str() );
    if( clusterSize < 100 || clusterSize > 10000 )
    {
        logStream << "Invalid clusterSize value in input file.\n";
        throwError( "Invalid clusterSize value in input file." );
        return false;
    }

    // extract descriptorSize
    pos = row.find( "descriptorsSize: " );
    end = row.find( ",", pos );
    if( pos==string::npos || end==string::npos )
    {
        logStream << "No descriptorSize specified in input file.\n";
        throwError( "Input file error." );
        return false;
    }
    descriptorSize = atoi( row.substr( pos+17, end ).c_str() );
    if( descriptorSize < 64 || descriptorSize > 128 )
    {
        logStream << "Invalid descriptorSize value in input file.\n";
        throwError( "Invalid descriptorSize value in input file." );
        return false;
    }

    return true;
}

void TrainWorker::readCodebook( fstream &fin, cv::Mat &codebook )
{
    string row;
    char*  pch;
    int    rowcounter = 0;

    vector< vector< float > > matrow;
    matrow.resize( clusterSize );

    getline( fin, row );
    while( !fin.eof() )
    {
        pch = strtok( (char*)row.c_str(), "|" );

        while( pch != NULL )
        {
            matrow[ rowcounter ].push_back( strtofloat( pch ) );
            pch = strtok( NULL, "|" );
        }
        rowcounter++;

        getline( fin, row );
    }

    for( int i=0; i<codebook.rows; i++ )
        for( int j=0; j<codebook.cols; j++ )
            codebook.at<float>(i,j) = matrow.at(i).at(j);
}
