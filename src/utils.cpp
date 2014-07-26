#include "utils.h"

#include <sstream>
#include <stdio.h>
#include <algorithm>

LogStream logStream;
string    homePath = "";

bool alphabeticalSort( string a, string b )
{
    std::transform( a.begin(), a.end(), a.begin(), ::tolower );
    std::transform( b.begin(), b.end(), b.begin(), ::tolower );
    return a.compare( b ) < 0;
}

int num_dirs( const char* path, vector< string >& dir_list )
{
    int dir_count = 0;
    struct dirent* dent;
    DIR* srcdir = opendir( path );

    if( srcdir == NULL )
        return -1;

    while( ( dent = readdir( srcdir ) ) != NULL )
    {
        struct stat st;

        if( strcmp( dent->d_name, "." ) == 0 || strcmp( dent->d_name, ".." ) == 0 )
            continue;

        if( fstatat( dirfd( srcdir ), dent->d_name, &st, 0 ) < 0 ) // not a directory
            continue;

        if( S_ISDIR( st.st_mode ) )
        {
            dir_count++;
            dir_list.push_back( dent->d_name );
        }
    }
    closedir( srcdir );

    std::sort( dir_list.begin(), dir_list.end(), alphabeticalSort );
    return dir_count;
}

int num_images( const char* path )
{
    vector< string > file_list; // will be discarded

    return num_images( path, file_list );
}

int num_images( const char* path, vector< string >& file_list )
{
    DIR * dirp;
    struct dirent * entry;
    int file_count = 0;

    dirp = opendir( path );

    // Error!
    if( dirp == NULL )
        return 0;

    while( ( entry = readdir( dirp ) ) != NULL )
    {
        if( entry->d_type != DT_REG ) // If the entry is a regular file
            continue;

        string filename = entry->d_name;
        string file_ext = filename.substr( filename.rfind(".") );

        for( unsigned int i = 0; i< file_ext.length(); i++ ) // convert extention to lowercase
            file_ext[ i ] = tolower( file_ext[ i ] );

        if( file_ext == ".png" || file_ext == ".jpg" )
        {
            file_count++;
            file_list.push_back( entry->d_name );
        }
    }

    closedir( dirp );

    std::sort( file_list.begin(), file_list.end(), alphabeticalSort );
    return file_count;
}

float strtofloat( const string& what )
{
    istringstream instr( what );
    float val;
    instr >> val;
    return val;
}

string toGlobalPath( string localPath )
{
    string globalPath = localPath;
    size_t pos        = localPath.find( homePath );
    
    if( !homePath.empty() && pos!=string::npos )
        globalPath = localPath.replace( pos, homePath.length(), "~" );
    
    return globalPath;
}

string toLocalPath( string globalPath )
{
    string localPath = globalPath;
    size_t pos       = globalPath.find( "~" );

    if( !homePath.empty() && pos!=string::npos )
        localPath = globalPath.replace( pos, 1, homePath );
    
    return localPath;
}

QString toGlobalPath( QString localPath  )
{
    return QString( toGlobalPath( localPath.toStdString() ).c_str() );
}

QString toLocalPath( QString globalPath )
{
    return QString( toLocalPath( globalPath.toStdString() ).c_str() );
}

time_t fileLastModification( const char* filename )
{
    struct stat st;

    if( stat( filename, &st )==0 )
        return st.st_mtime;

    return -1;
}
