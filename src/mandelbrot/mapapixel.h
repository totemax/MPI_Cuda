#ifndef _MAPAPIXEL_H_
#define _MAPAPIXEL_H_

//-------------------------------------------------------------------+
// PCM. Procesamiento Paralelo Curso 03/04 EUI             04/04/06  |
//                                                                   |
// mapapixel.h: Modulo que permite trabajar en modo grafico con GTK  |
//-------------------------------------------------------------------+

#define MAX_COLUMNAS       1280
#define MAX_FILAS          1024

typedef enum {
    escalaGrises,    // Cada pixel un byte. 0 => Negro y 255 => Blanco
    colorRGB         // Cada pixel tres bytes indicando RGB
} tipoColor;

typedef struct tRGB {
    unsigned char rojo;   // Negro  => RGB todos a 0
    unsigned char verde;  // Blanco => RGB todos a 255
    unsigned char azul;
} tipoRGB;

//-------------------------------------------------------------------+
// Prototipos de funciones necesarias para inicializar la ventana de |
// dibujo.                                                           |
//-------------------------------------------------------------------+
typedef void (*mapiFuncionAccion) (void);

typedef void (*mapiFuncionClick) (short columna, short fila,
    int botonIzquierdo);

    typedef void (*mapiFuncionCierre) (void);

    //-------------------------------------------------------------------+
    // Inicializar un mapa de pixels supondra la aparicion de una ventana|
    // con un area grafica de tamanio "filas" * "columnas" donde podra   |
    // dibujarse mediante funciones de dibujar punto, linea, etc. o bien |
    // representar de golpe una imagen guardada en un mapa en memoria    |
    // mediante la funcion "mapiDibujarMapa". Inicialmente, la ventana   |
    // grafica se mostrara en tonalidad gris.                            |
    //                                                                   |
    // Para permitir interactuar con la imagen representada, tambien     |
    // puede pasarse una funcion asociada al click del raton de forma    |
    // que se informa de la fila y columna donde se hizo click y si se   |
    // pulso el boton izquierdo del raton o no (el derecho).             |
    // Tambien pueden asociarse funciones al evento de pulsar el boton   |
    // de accion y de cierre de la venta.                                |
    //-------------------------------------------------------------------+
    int mapiInicializar (short filas, short columnas,
        mapiFuncionAccion fAccion,
        mapiFuncionClick  fClick,
        mapiFuncionCierre fCierre);

        //-------------------------------------------------------------------+
        // Puede definirse una gama de colores mediante la profundidad       |
        // (expresada en numero de bits) de cada color RGB. Por ejemplo, si  |
        // se indica profundidad 3, se dispondra de 512 colores.             |
        // Por defecto, la profundidad del color es 8 (color verdadero).     |
        //-------------------------------------------------------------------+
        void mapiProfundidadColor (unsigned short profundidad);

        int  mapiNumColoresDefinidos (void);

        void mapiColorRGB (int color, tipoRGB *colorRGB);

        void mapiDibujarPunto (short fila,  short columna,  int color);

        void mapiDibujarLinea (short fila1, short columna1, short fila2, short columna2, int color);

        void mapiDibujarRectangulo (short fila,  short columna, short ancho, short largo, int color);

        //-------------------------------------------------------------------+
        // Para poder leer y grabar imagenes en el sistema de archivos, se   |
        // utilizara una estructura de "tipoMapa". Estas estructuras podran  |
        // manipularse por el usuario y representarse en la ventana grafica. |
        //-------------------------------------------------------------------+
        typedef struct tMapa {
            tipoColor elColor;
            short     filas;
            short     columnas;
            char      *pixels;   // Puntero a un array bidimensional de
            // dimensiones [filas *  columnas ] de un byte
            // o tres segun sea o no elColor escalaGrises
        } tipoMapa;

        //-------------------------------------------------------------------+
        // Con la funcion siguiente, se obtiene memoria para los pixels del  |
        // mapa pasado como primer parametro y todos los pixels se ponen a   |
        // negro.                                                            |
        //-------------------------------------------------------------------+
        void mapiCrearMapa (tipoMapa *unMapa);

        //-------------------------------------------------------------------+
        // Con la siguiente funcion, puede leerse una imagen en formato "PGM"|
        // binario => escalaGrises o formato "PPM" binario => colorRGB       |
        // La lectura de la imagen se encarga de pedir la memoria necesaria, |
        // pero no se dibuja en la ventana grafica.                          |
        //-------------------------------------------------------------------+
        void mapiLeerMapa (char *fichero, tipoMapa *unMapa);

        //-------------------------------------------------------------------+
        // Las dos funciones siguientes permiten poner un pixel de un mapa al|
        // valor que se desee. El cambio en la imagen representada en el mapa|
        // no se visualizara en la ventana grafica hasta que no se invoque la|
        // representacion del mapa completo.                                 |
        //-------------------------------------------------------------------+
        void mapiPonerPuntoGris   (tipoMapa *unMapa, short fila, short columna, short tonalidad);

        void mapiPonerPuntoRGB    (tipoMapa *unMapa, short fila, short columna, tipoRGB color);

        //-------------------------------------------------------------------+
        // Puede dibujarse un mapa en la ventana grafica, pero para ello,    |
        // debe haberse previamente inicializado con las mismas dimensiones  |
        // (filas y columnas).                                               |
        //-------------------------------------------------------------------+
        void mapiDibujarMapa (tipoMapa *unMapa);

        //-------------------------------------------------------------------+
        // Para grabar en disco el mapa indicado.                            |
        //-------------------------------------------------------------------+
        void mapiGrabarMapa (char *fichero, tipoMapa *unMapa);
        #endif
