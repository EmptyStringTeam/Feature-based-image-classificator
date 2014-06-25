#include "logstream.h"

#include <ios>
#include <ctime>

using namespace std;

LogStream::LogStream():
    logToFile(false)
{
}

LogStream::~LogStream()
{
    if( logFile.is_open() )
    {
        logFile.flush();
        logFile.close();
    }
}

bool LogStream::Init( const char* filename /*= "logfile.log"*/, bool logToFile/* = true*/ )
{
    this->logToFile = logToFile;
    if( logToFile )
    {
        logFile.open( filename, ios::out | ios::app );

        if( logFile.is_open() )
        {
            time_t now = time ( 0 );
            char*  dt  = ctime( &now );

            logFile << "\n\n----------- Logfile: " << dt <<endl;
            cout << "Logfile \"" << filename << "\" succesfully opened." << endl;
        }
        else
        {
            cerr<< "Unable to open log file \"" << filename << "\"!" << endl;
            logToFile = false;
        }
    }

    return logFile.is_open();
}

void LogStream::flush()
{
    cout.flush();
    if( logToFile )
        logFile.flush();
}
