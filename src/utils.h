#ifndef UTILS_H
#define UTILS_H

#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string>
#include <cstring>
#include <vector>
#include <qt5/QtCore/qstring.h>
#include "logstream.h"

using namespace std;

int num_dirs( const char* path, vector< string >& dir_list );

int num_images( const char* path );
int num_images( const char* path, vector< string >& file_list );

float strtofloat( const string& what );

extern LogStream logStream;
extern string    homePath;

string toGlobalPath( string localPath  );
string toLocalPath ( string globalPath );

QString toGlobalPath( QString localPath  );
QString toLocalPath ( QString globalPath );

time_t fileLastModification(const char* filename );

#endif // UTILS_H
