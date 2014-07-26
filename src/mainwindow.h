#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QAction>
#include <QLabel>
#include <QProgressBar>
#include <QToolButton>
#include <QPushButton>
#include <opencv2/opencv.hpp>
#include <time.h>

using namespace std;


namespace Ui {
    class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow( QWidget *parent = 0 );
    ~MainWindow();

private:
    QLabel*       statusLabel;
    QLabel*       inputCodebookLabel;
    QProgressBar* progressBar;
    QToolButton*  abortBtn;
    QPushButton*  audioToggleBtn;

    QObject*        controller;
    struct timespec progressBarTimer;

    void enableControls( bool enable );

    bool audioMuted;
    int  audioMutedNum;
    void playSuccess();
    void playError();

    string datasetFolder;
    string datasetName;
    string method;
    string histogramMethod;
    int    splitPercent;
    int    minHessian;
    int    clusterSize;
    int    descriptorSize;
    int    datasetSize;
    time_t lastModified;

    bool abortRequested;

private slots:
    void on_codebookBtn_clicked();
    void on_trainBtn_clicked   ();

    void on_inputFolderBtn_clicked   ();
    void on_outputCodebookBtn_clicked();
    void on_inputCodebookBtn_clicked ();

    void on_splitSlider_valueChanged();
    void on_featureDetectionMethod_currentIndexChanged(int);

    void on_inputCodebook_textChanged ();
    void on_outputCodebook_textChanged();

    void toggleAudio();
    void abortOperation();

    void updateCodebookLabel();
    bool readCodebookMetadata( fstream& );

public:
    Ui::MainWindow *ui;

public slots:
    // Codebook creation
    void codebook_imgReadingDone();
    void codebook_clusteringDone();

    // SVM training
    void training_imgReadingDone();
    void training_trainingDone  ();
    void training_testingDone   ( float accuracy );

    // common signals
    void processingDone   ();
    void setupProgressBar ( QString inputFolder, int splitPercent );
    void updateProgressbar();
    void throwError       ( const QString& error );
    void checkAbort       ( bool* abortRequested );
};


#endif // MAINWINDOW_H


