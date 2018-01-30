//-------------------------------------------------------------------+
// PCM. Procesamiento Paralelo  Curso 03/04 EUI            04/04/06  |
//                                                                   |
// mapapixel.c: Modulo que permite trabajar en modo grafico con GTK  |
//-------------------------------------------------------------------+

#include <sys/stat.h>
#include <fcntl.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <gtk/gtk.h>

#include "mapapixel.h"

#define MAX_LONG_LINEA 128
#define DELTA_COLOR    256

static GtkWidget *window;
static GtkWidget *pizarra;
static GtkWidget *vbox;
static GtkWidget *button;
static GdkGC     *miGC;

static int laProfundidad    = 8;
static int elDesplazamiento = 0;
static int elNumColores     = 16777216;
static int mascara          = 0xFF;
static int ultimoColor      = -1;
static int colorVerdadero   = TRUE;

static mapiFuncionClick  userfClick;
static mapiFuncionCierre userfCierre;

//--------------------------------------------------------------------
void leerLinea (int fd, char *linea) {
    int i;

    for (i=0; i<MAX_LONG_LINEA; i++) {
        assert (read (fd, &linea[i], 1) == 1);
        if (linea[i] == '\n') {
            linea[i] = 0;
            return;
        }
    }
    assert (0);
}

//--------------------------------------------------------------------
static gint clickRaton (GtkWidget      *widget,
    GdkEventButton *evento)
    {
        int columna, fila;
        GdkModifierType estado;

        if ( (evento->button == 1) || (evento->button == 3)) {
            gdk_window_get_pointer (evento->window, &columna, &fila, &estado);
            userfClick (fila, columna, (evento->button == 1));
        }
        return 0;
    }

    //--------------------------------------------------------------------
    static void quit ()
    {
        if (userfCierre != NULL) userfCierre();
        gtk_exit (0);
    }

    //--------------------------------------------------------------------
    static void ponerColor (int color) {
        tipoRGB  cRGB;
        GdkColor elColor;

        if (ultimoColor != color) {
            if (colorVerdadero) {
                mapiColorRGB (color, &cRGB);
                elColor.pixel = (cRGB.rojo*65536 + cRGB.verde*256 + cRGB.azul);
            } else
            elColor.pixel = (color % 256);
            gdk_gc_set_foreground (miGC, &elColor);
            ultimoColor = color;
        }
    }

    //    FUNCIONES EXPORTADAS

    //--------------------------------------------------------------------
    int mapiInicializar (short filas, short columnas,
        mapiFuncionAccion fAccion,
        mapiFuncionClick  fClick,
        mapiFuncionCierre fCierre)
        {
            gtk_init (NULL, NULL);
            gdk_rgb_init();


            window = gtk_window_new (GTK_WINDOW_TOPLEVEL);

            vbox = gtk_vbox_new (FALSE, 0);
            gtk_container_add (GTK_CONTAINER (window), vbox);
            gtk_widget_show (vbox);

            gtk_signal_connect (GTK_OBJECT (window), "destroy",
            GTK_SIGNAL_FUNC (quit), NULL);

            /* Crear area de dibujo */
            pizarra = gtk_drawing_area_new ();
            gtk_drawing_area_size (GTK_DRAWING_AREA (pizarra), columnas, filas);
            gtk_box_pack_start (GTK_BOX (vbox), pizarra, TRUE, TRUE, 0);

            gtk_widget_show (pizarra);

            // Seniales de eventos
            if (fClick != NULL) {
                userfClick = fClick;
                gtk_signal_connect (GTK_OBJECT (pizarra), "button_press_event",
                (GtkSignalFunc) clickRaton, NULL);
                gtk_widget_set_events (pizarra, GDK_BUTTON_PRESS_MASK);
            }

            userfCierre = fCierre;

            // Boton de inicio del dibujo
            button = gtk_button_new_with_label ("Accion");
            gtk_box_pack_start (GTK_BOX (vbox), button, FALSE, FALSE, 0);

            gtk_signal_connect_object (GTK_OBJECT (button), "clicked",
            GTK_SIGNAL_FUNC (fAccion),
            NULL);
            gtk_widget_show (button);
            gtk_widget_show (window);

            // Comprobar que el tipo de visual es color verdadero
            if (! (gdk_window_get_visual(pizarra->window)->type == GDK_VISUAL_TRUE_COLOR)) {
                printf ("El controlador grafico deberia estar en modo color verdadero\n");
                colorVerdadero = FALSE;
            }

            miGC = gdk_gc_new (pizarra->window);
            gtk_main ();

            return 0;
        }

        //--------------------------------------------------------------------
        void mapiProfundidadColor (unsigned short profundidad) {
            int i;

            laProfundidad    = profundidad;
            elDesplazamiento = 8 - profundidad;
            elNumColores  = 1 << (profundidad*3);
            mascara = 1;
            for (i=1; i<profundidad; i++) mascara  = (mascara  << 1) | 1;
        }

        //--------------------------------------------------------------------
        int mapiNumColoresDefinidos (void) {
            return elNumColores;
        }

        //--------------------------------------------------------------------
        void mapiColorRGB (int color, tipoRGB *colorRGB) {
            colorRGB->rojo  = (color & mascara ) << elDesplazamiento;
            color = color >> (laProfundidad - 1);
            colorRGB->verde = (color & mascara ) << elDesplazamiento;
            color = color >> (laProfundidad - 1);
            colorRGB->azul  = (color & mascara ) << elDesplazamiento;
        }

        //--------------------------------------------------------------------
        void mapiDibujarPunto (short fila, short columna, int color) {
            ponerColor (color);
            gdk_draw_point (pizarra->window, miGC, columna, fila);
        }

        //--------------------------------------------------------------------
        void mapiDibujarLinea (short fila1, short columna1,
            short fila2, short columna2, int color) {

                ponerColor (color);
                gdk_draw_line (pizarra->window, miGC, columna1, fila1, columna2, fila2);
            }

            //--------------------------------------------------------------------
            void mapiDibujarRectangulo (short fila1, short columna1,
                short ancho, short largo, int color) {

                    ponerColor (color);
                    gdk_draw_rectangle (pizarra->window, miGC, TRUE, columna1, fila1,
                        ancho, largo);
                    }

                    //--------------------------------------------------------------------
                    void mapiCrearMapa (tipoMapa *unMapa)
                    {
                        int numBytes;

                        assert (   (unMapa->filas <= MAX_FILAS)
                        && (unMapa->columnas <= MAX_COLUMNAS));
                        numBytes = unMapa->filas * unMapa->columnas;
                        if (unMapa->elColor == colorRGB) numBytes = numBytes * 3;
                        unMapa->pixels = malloc (numBytes);
                        bzero (unMapa->pixels, numBytes);
                    }

                    //--------------------------------------------------------------------
                    void mapiLeerMapa (char *fichero, tipoMapa *unMapa)
                    {
                        int fd, numBytes, intColumnas, intFilas;
                        char linea[MAX_LONG_LINEA];

                        if (*fichero != '/') {
                            strcpy (linea, getenv("PWD"));
                            strcat (linea, "/");
                            strcat (linea, fichero);
                        } else
                        strcpy (linea, fichero);

                        assert ((fd = open (linea, O_RDONLY)) != -1);
                        leerLinea(fd, linea);
                        if      (strcmp (linea, "P5") == 0)
                        unMapa->elColor = escalaGrises;
                        else if (strcmp (linea, "P6") == 0)
                        unMapa->elColor = colorRGB;
                        else
                        assert (0);
                        leerLinea(fd, linea);
                        assert (sscanf (linea, "%d %d", &intColumnas, &intFilas) == 2);
                        assert ((intFilas <= MAX_FILAS) && (intColumnas <= MAX_COLUMNAS));
                        unMapa->filas    = intFilas;
                        unMapa->columnas = intColumnas;
                        leerLinea(fd, linea);
                        numBytes = unMapa->filas * unMapa->columnas;
                        if (unMapa->elColor == colorRGB) numBytes = numBytes * 3;
                        unMapa->pixels = malloc (numBytes);
                        assert (read (fd, unMapa->pixels, numBytes) == numBytes);
                        assert (close(fd) == 0);
                    }

                    //--------------------------------------------------------------------
                    void mapiPonerPuntoGris (tipoMapa *unMapa,
                        short fila, short columna, short tonalidad) {
                            guchar *elPixel = (guchar *) unMapa->pixels;

                            elPixel = elPixel + (fila * unMapa->columnas) + columna;
                            *elPixel = tonalidad;
                        }

                        //--------------------------------------------------------------------
                        void mapiPonerPuntoRGB  (tipoMapa *unMapa,
                            short fila, short columna, tipoRGB color) {
                                guchar *elPixel = (guchar *) unMapa->pixels;

                                elPixel = elPixel + (((fila * unMapa->columnas) + columna) * 3);
                                *elPixel++ = color.rojo;
                                *elPixel++ = color.verde;
                                *elPixel++ = color.azul;
                            }

                            //--------------------------------------------------------------------
                            void mapiDibujarMapa (tipoMapa *unMapa) {
                                if (unMapa->elColor == escalaGrises)
                                gdk_draw_gray_image (pizarra->window,
                                    pizarra->style->fg_gc[GTK_STATE_NORMAL],
                                    0, 0, unMapa->columnas, unMapa->filas,
                                    GDK_RGB_DITHER_NONE, (guchar *) unMapa->pixels,
                                    unMapa->columnas);
                                    else
                                    gdk_draw_rgb_image  (pizarra->window,
                                        pizarra->style->fg_gc[GTK_STATE_NORMAL],
                                        0, 0, unMapa->columnas, unMapa->filas,
                                        GDK_RGB_DITHER_MAX, (guchar *) unMapa->pixels,
                                        unMapa-> columnas * 3);

                                    }

                                    //--------------------------------------------------------------------
                                    void mapiGrabarMapa (char *fichero, tipoMapa *unMapa)
                                    {
                                        int  fd, numBytes;
                                        char linea[MAX_LONG_LINEA];

                                        if (*fichero != '/') {
                                            strcpy (linea, getenv("PWD"));
                                            strcat (linea, "/");
                                            strcat (linea, fichero);
                                        } else
                                        strcpy (linea, fichero);
                                        assert ((fd = open (linea, O_CREAT | O_WRONLY, S_IRWXU)) != -1);
                                        if (unMapa->elColor == escalaGrises) {
                                            numBytes = unMapa->filas * unMapa->columnas;
                                            sprintf (linea, "P5\n");
                                        } else {
                                            numBytes = unMapa->filas * unMapa->columnas * 3;
                                            sprintf (linea, "P6\n");
                                        }
                                        assert (write (fd, linea, 3) == 3);
                                        assert (sprintf(linea, "%4d %4d\n", unMapa->columnas, unMapa->filas) == 10);
                                        assert (write (fd, linea, 10) == 10);
                                        assert (sprintf(linea, "255\n") == 4);
                                        assert (write (fd, linea, 4) == 4);
                                        assert (write (fd, unMapa->pixels, numBytes) == numBytes);
                                        assert (close(fd) == 0);
                                    }
