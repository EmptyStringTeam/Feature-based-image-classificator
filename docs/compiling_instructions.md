<big>**Feature-based image detection project: compiling instructions**</big>

<br />

**Compiling OpenCV with Kaze support**

*<small>Adapted from <a href="http://www.raben.com/book/export/html/3" target="_blank">http://www.raben.com/book/export/html/3</a></small>*

Note: all apt-get install commands here use -y, which will "Assume Yes to all queries and do not prompt", not showing  download size and disk space needed to install.

&rarr; *Install dependencies*

sudo apt-get -y install build-essential cmake pkg-config

sudo apt-get -y install libjpeg62-dev

sudo apt-get -y install libtiff4-dev libjasper-dev

sudo apt-get -y install libgtk2.0-dev

sudo apt-get -y install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev

sudo apt-get -y install python-dev python-numpy

sudo apt-get -y install libtbb-dev

sudo apt-get -y install libqt4-dev

&rarr; *Install Qt creator*

sudo apt-get -y install qtcreator

&rarr; *Download BloodAxe's fork of OpenCV with Kaze support*

wget https://github.com/BloodAxe/opencv/archive/kaze.zip

unzip kaze.zip

&rarr; *Compile*

cd opencv-kaze

mkdir build

cd build

cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_TBB=ON -D BUILD_NEW_PYTHON_SUPPORT=ON -D WITH_V4L=ON -D WITH_QT=ON -D WITH_OPENGL=ON ./..
    
sudo make install

<br/>

**Setup VLFeat**

&rarr; *Download and extract VLFeat library*

wget http://www.vlfeat.org/download/vlfeat-0.9.18-bin.tar.gz

tar -xvf vlfeat-0.9.18-bin.tar.gz

cd vlfeat-0.9.18

&rarr; *Add header files to include path*

sudo mkdir /usr/local/include/vl	

sudo cp vl/*.h /usr/local/include/vl/

&rarr; *Add shared library to bin path*

sudo mkdir /usr/local/lib/vl

[*for 64 bit systems*]
	sudo cp bin/glnxa64/libvl.so /usr/local/lib/vl/
	
[*for 32 bit systems*] 
	sudo cp bin/glnx86/libvl.so /usr/local/lib/vl/

&rarr; *Register library with ldconfig*

sudo su

echo "/usr/local/lib/vl" > /etc/ld.so.conf.d/vl.conf

ldconfig

exit

<br/>

**Compile and run**

Open the project file *Feature-based_image_classificator.pro*, setup compile paths and run!

<br/>





