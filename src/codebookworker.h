#ifndef CODEBOOKWORKER_H
#define CODEBOOKWORKER_H

#include <QThread>

using namespace std;

class CodebookWorker : public QObject
{
     Q_OBJECT
     QThread workerThread;
     bool    abortRequested;

 public:
     CodebookWorker();

 public slots:
     void doWork( QString method, QString inputFolder, QString outputFile, int splitPercent, int minHessian, int clusterSize );

 signals:
     void imgReadingDone();
     void clusteringDone();

     // common signals
     void processingDone   ();
     void setupProgressBar ( QString inputFolder, int splitPercent );
     void updateProgressbar();
     void throwError       ( const QString& error );
     void checkAbort       ( bool* abortRequested );
};

#endif // CODEBOOKWORKER_H
