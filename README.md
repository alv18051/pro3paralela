# pro3paralela
Proyecto no. 3 Computación Paralela

1. Instalar las dependencias con "./install_packages.sh"
2. ejecutar el comando nvcc -o houghBase houghBase.cu $(pkg-config --cflags --libs opencv4) -lcuda -lcudart para compilar el programa con memoria global
3. ejecutar el comando nvcc -o houghMod houghMod.cu $(pkg-config --cflags --libs opencv4) -lcuda -lcudart para compilar el programa con memoria constante
4. ejecutar el comando nvcc -o houghShd houghModSH.cu $(pkg-config --cflags --libs opencv4) -lcuda -lcudart para compilar el programa con memoria compartida
5. Correr el ejecutable con ./houghBase nombre_del_archivo.pgm (Importante que el archivo este en el mismo directorio del proyecto y sea .pgm)
6. Correr el ejecutable con ./houghMod nombre_del_archivo.pgm (Importante que el archivo este en el mismo directorio del proyecto y sea .pgm)
7. Correr el ejecutable con ./houghShd nombre_del_archivo.pgm (Importante que el archivo este en el mismo directorio del proyecto y sea .pgm)
