#include "traincontroller.h"

#include "trainworker.h"

TrainController::TrainController( MainWindow* caller, QString inputFile, QString histogramMethod )
{
    TrainWorker *worker = new TrainWorker();
    worker->moveToThread( &workerThread );
    connect( &workerThread, SIGNAL( finished() ), worker, SLOT( deleteLater() ) );
    connect( this, SIGNAL( operate(QString,QString) ), worker, SLOT( doWork(QString,QString) ) );

    connect( worker, SIGNAL( imgReadingDone()   ), caller, SLOT( training_imgReadingDone()   ) );
    connect( worker, SIGNAL( trainingDone()     ), caller, SLOT( training_trainingDone()     ) );
    connect( worker, SIGNAL( testingDone(float) ), caller, SLOT( training_testingDone(float) ) );

    // common signals
    connect( worker, SIGNAL( processingDone()              ), caller, SLOT( processingDone()              ) );
    connect( worker, SIGNAL( setupProgressBar(QString,int) ), caller, SLOT( setupProgressBar(QString,int) ) );
    connect( worker, SIGNAL( updateProgressbar()           ), caller, SLOT( updateProgressbar()           ) );
    connect( worker, SIGNAL( throwError(QString)           ), caller, SLOT( throwError(QString)           ) );

    //abort signal
    connect( worker, SIGNAL( checkAbort(bool*) ), caller, SLOT( checkAbort(bool*) ), Qt::BlockingQueuedConnection );

    workerThread.start();

    this->operate( inputFile, histogramMethod );
}

TrainController::~TrainController()
{
    if( workerThread.isRunning() )
    {
        workerThread.terminate();
        workerThread.wait();
    }
}
