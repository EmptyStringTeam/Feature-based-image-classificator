#ifndef CODEBOOKCONTROLLER_H
#define CODEBOOKCONTROLLER_H

#include <QThread>
#include "mainwindow.h"

class CodebookController : public QObject
{
     Q_OBJECT
     QThread workerThread;

 public:
     CodebookController ( MainWindow* caller, QString method, QString inputFolder, QString outputFile, int splitPercent, int minHessian, int clusterSize );
     ~CodebookController();

signals:
     void operate( QString, QString, QString, int, int, int );
};

#endif // CODEBOOKCONTROLLER_H
