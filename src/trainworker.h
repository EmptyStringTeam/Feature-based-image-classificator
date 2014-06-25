#ifndef TRAINWORKER_H
#define TRAINWORKER_H

#include <QThread>
#include <opencv2/opencv.hpp>

extern "C"
{
    #include <vl/gmm.h>
}

using namespace std;

class TrainWorker : public QObject
 {
     Q_OBJECT
     QThread workerThread;

public:
     TrainWorker();

private:
     string datasetFolder;
     string method;
     string histogramMethod;
     int    splitPercent;
     int    minHessian;
     int    clusterSize;
     int    descriptorSize;
     int    datasetSize;
     vector< string > categories;

     VlGMM* gmm;

public slots:
     void doWork( QString inputFile, QString histogramMethod );

private slots:
     bool readCodebookMetadata( fstream &fin );
     void readCodebook        ( fstream &fin, cv::Mat &codebook );

     void setupGMM( const cv::Mat &codebook );

     vector< float > createHistogram      ( const cv::Mat &codebook,    const cv::Mat &descriptors );
     vector< float > createBOWHistogram   ( const cv::Mat &codebook,    const cv::Mat &descriptors );
     vector< float > createFisherHistogram( const cv::Mat &descriptors, int dimension );

     float accuracy ( const vector< vector< int > > &confusion );
     float precision( const vector< int > col, int index );
     float recall   ( const vector< int > row, int index );
     void  createConfusionMatrixImg( const vector< vector< int > > &confusion );

 signals:
     void imgReadingDone();
     void trainingDone  ();
     void testingDone   ( float errRate );

     // common signals
     void processingDone   ();
     void setupProgressBar ( QString inputFolder, int splitPercent );
     void updateProgressbar();
     void throwError       ( const QString& error );
     void checkAbort       ( bool* abortRequested );
};

#endif // TRAINWORKER_H
