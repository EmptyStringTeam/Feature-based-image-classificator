#include "codebookcontroller.h"

#include "codebookworker.h"

CodebookController::CodebookController( MainWindow* caller, QString method, QString inputFolder, QString outputFile, int splitPercent, int minHessian, int clusterSize )
{
    CodebookWorker *worker = new CodebookWorker();
    worker->moveToThread( &workerThread );
    connect( &workerThread, SIGNAL( finished() ), worker, SLOT( deleteLater() ) );
    connect( this, SIGNAL( operate(QString,QString,QString,int,int,int) ), worker, SLOT( doWork(QString,QString,QString,int,int,int) ) );

    connect( worker, SIGNAL( imgReadingDone() ), caller, SLOT( codebook_imgReadingDone() ) );
    connect( worker, SIGNAL( clusteringDone() ), caller, SLOT( codebook_clusteringDone() ) );
    // common signals
    connect( worker, SIGNAL( processingDone()              ), caller, SLOT( processingDone()              ) );
    connect( worker, SIGNAL( setupProgressBar(QString,int) ), caller, SLOT( setupProgressBar(QString,int) ) );
    connect( worker, SIGNAL( updateProgressbar()           ), caller, SLOT( updateProgressbar()           ) );
    connect( worker, SIGNAL( throwError(QString)           ), caller, SLOT( throwError(QString)           ) );

    //abort signal
    connect( worker, SIGNAL( checkAbort(bool*) ), caller, SLOT( checkAbort(bool*) ), Qt::BlockingQueuedConnection );

    workerThread.start();

    this->operate( method, inputFolder, outputFile, splitPercent, minHessian, clusterSize );
}

CodebookController::~CodebookController()
{
    workerThread.terminate();
    workerThread.wait();
}
