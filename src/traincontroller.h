#ifndef TRAINCONTROLLER_H
#define TRAINCONTROLLER_H

#include <QThread>
#include "mainwindow.h"

class TrainController : public QObject
 {
     Q_OBJECT
     QThread workerThread;

public:
     TrainController ( MainWindow* caller, QString inputFile, QString histogramMethod );
     ~TrainController();

signals:
     void operate( QString, QString );
 };

#endif // TRAINCONTROLLER_H
