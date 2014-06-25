#ifndef LOGSTREAM_H
#define LOGSTREAM_H

#include <ostream>
#include <fstream>
#include <iostream>
using namespace std;

class LogStream
{
public:
    LogStream();
    ~LogStream();

    bool Init( const char* filename = "logfile.log", bool logToFile = true );

public:
    template< typename T >
    LogStream& operator<< (T obj);

    void flush();

private:
    ofstream logFile;
    bool     logToFile;
};

template< typename T >
inline LogStream& LogStream::operator <<(T value)
{
    cout << value;
    cout.flush();
    if( logToFile )
        logFile << value;

    return *this;
}

template< >
inline LogStream& LogStream::operator <<(string value)
{
    cout << value.c_str();
    cout.flush();
    if( logToFile )
        logFile << value.c_str();

    return *this;
}

template< >
inline LogStream& LogStream::operator <<(const char* value)
{
    cout << value;
    cout.flush();
    if( logToFile )
        logFile << value;

    return *this;
}

template< >
inline LogStream& LogStream::operator <<(ostream& value)
{
    cout << value;
    cout.flush();
    if( logToFile )
        logFile << value;

    return *this;
}

#ifdef _GLIBCXX_USE_WCHAR_T
template< >
inline LogStream& LogStream::operator <<(wostream& value)
{
    cout << value;
    cout.flush();
    if( logToFile )
        logFile << value;

    return *this;
}
#endif // _GLIBCXX_USE_WCHAR_T

#endif // LOGSTREAM_H
