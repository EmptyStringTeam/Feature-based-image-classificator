#include "codebookworker.h"

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
#include <omp.h>
#include <stdlib.h>

#include "utils.h"

using namespace std;
using namespace cv;

CodebookWorker::CodebookWorker()
{
}

void CodebookWorker::doWork( QString method, QString inputFolder, QString outputFile, int splitPercent, int minHessian, int clusterSize )
{
    vector< string > dir_list;
    struct timespec start, img_read, clustering;

    logStream << "\nCODEBOOK -- using method: " << method.toStdString();
    if( method == "SURF" )
        logStream << ", minHessian: " << minHessian;
    logStream << ", clusterSize: " << clusterSize << "\n";

    Mat codebook;

    Feature2D *detector = NULL;
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

    if( detector == NULL || extractor == NULL)
    {
        logStream << "Initialization error, aborting.\n";
        throwError( method + " initialization error" );
        return;
    }

    if( num_dirs( inputFolder.toStdString().c_str(), dir_list ) < 0 )
    {
        logStream << "Incorrect input folder, aborting.\n";
        throwError( "Incorrect input folder" );
        return;
    }

    clock_gettime( CLOCK_REALTIME, &start );

    int totalKeypoints  = 0;
    int totalFilesCount = 0;

    emit setupProgressBar( inputFolder, splitPercent );

    vector< vector < float > > featuresMat;

    // Create file list using splitPercent files of each directory
    vector < pair < int, string > > fileList;
    for( unsigned int dirIx = 0; dirIx<dir_list.size(); dirIx++ )
    {
        char path[256];
        sprintf( path,"%s/%s", inputFolder.toStdString().c_str(), dir_list[ dirIx ].c_str() );

        vector< string > pathFileList;

        int dirFileCount   = num_images( path, pathFileList );
        int trainFileCount = floor( dirFileCount * splitPercent / 100.0f );

        for( int fileIx = 0; fileIx < trainFileCount; fileIx++ )
        {
            sprintf( path,"%s/%s/%s", inputFolder.toStdString().c_str(), dir_list[ dirIx ].c_str(), pathFileList[ fileIx ].c_str() );
            fileList.push_back( pair<int,string>( dirIx, path ) );
        }
    }

    #pragma omp parallel for shared(detector,extractor,totalFilesCount,featuresMat)
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

        // Cut points which normalization goes above 0.2 to reduce error given by luminosity -- SIFT only
        if( method == "SIFT" )
        {
            for( int i = 0; i<descriptors.rows; i++ )
            {
                long double sum = 0;

                // l2 norm
                for( int j = 0; j<descriptors.cols; j++ )
                    sum += (descriptors.at<float>(i,j)*descriptors.at<float>(i,j));

                sum = sqrt(sum);

                for( int j=0; j<descriptors.cols; j++ )
                    descriptors.at<float>(i,j) /= sum;

                // cut values above 0.2
                for( int j = 0; j<descriptors.cols; j++ )
                    descriptors.at<float>(i,j) = min( 0.2f, descriptors.at<float>(i,j) );

                // l2 norm
                sum = 0;

                for( int j=0; j<descriptors.cols; j++ )
                    sum += (descriptors.at<float>(i,j)*descriptors.at<float>(i,j));

                sum = sqrt(sum);

                for( int j=0; j<descriptors.cols; j++ )
                    descriptors.at<float>(i,j) /= sum;
            }
        }

        #pragma omp critical
        {
            for( int i=0; i<descriptors.rows; i++ )
            {
                vector< float > tmp;
                for( int j = 0; j<descriptors.cols; j++)
                    tmp.push_back( descriptors.at<float>(i,j) );

                featuresMat.push_back( tmp );
            }

            totalKeypoints += keypoints.size();
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

    // done reading images
    emit imgReadingDone();
    clock_gettime( CLOCK_REALTIME, &img_read );

    logStream << "\n";
    logStream << "Extracted " << totalKeypoints << " keypoints from " << totalFilesCount << " images\n";
    if( totalKeypoints < clusterSize )
    {
        clusterSize = totalKeypoints;
        logStream << "\nWarning: totalKeypoints is less than desired clusterSize!\nNew clusterSize is " << totalKeypoints << "\n\n";
    }

    int descriptorLen = method == "SIFT" ? 128 : 64;

    Mat data( featuresMat.size(), descriptorLen, CV_32F );
    for( unsigned int s = 0; s<featuresMat.size(); s++ )
        for( int i=0; i<descriptorLen; i++ )
            data.at<float>(s, i) = featuresMat[s][i];

    // do cluster
    Mat clusterLabels;
    Mat centers( clusterSize, descriptorLen, CV_32F );
    kmeans( data, clusterSize, clusterLabels, TermCriteria( CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 20, 0.0001 ), 20, KMEANS_PP_CENTERS, centers );

    codebook = centers;

    clock_gettime( CLOCK_REALTIME, &clustering );

    // save codebook file
    ofstream fout;

    fout.open( outputFile.toStdString().c_str(), ios::out );
    if( !fout.is_open() )
    {
        logStream << "Error writing codebook file.\n";
        throwError( "Error writing codebook file." );
        return;
    }

    // convert inputFolder from localPath to globalPath ( replace homePath with tilde )
    inputFolder = toGlobalPath( inputFolder );

    // First row: metadata
    fout << "Feature-detection method: " << method.toStdString();
    if( method == "SURF" )
        fout << ", minHessian: " << minHessian;
    fout << ", clusterSize: " << codebook.rows << ", descriptorsSize: " << codebook.cols;
    fout << ", dataset: '" << inputFolder.toStdString() << "', training percent: " << splitPercent << '%' << endl;

    for( int i = 0; i<codebook.rows; i++ )
    {
        for( int j = 0; j<codebook.cols; j++ )
            fout << codebook.at<float>(i,j) << "|";
        fout << endl;
    }
    fout.close();

    // print times
    int imgReadingMins = ( img_read.tv_sec - start.tv_sec ) / 60;
    int imgReadingSecs = ( img_read.tv_sec - start.tv_sec ) % 60;
    int clusteringMins = ( clustering.tv_sec - img_read.tv_sec ) / 60;
    int clusteringSecs = ( clustering.tv_sec - img_read.tv_sec ) % 60;

    logStream << "Image Reading:   " << imgReadingMins << "m, " << imgReadingSecs << "s.\n";
    logStream << "Clustering:      " << clusteringMins << "m, " << clusteringSecs << "s.\n";

    emit clusteringDone();
    emit processingDone();
}

