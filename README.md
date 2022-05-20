# TFG: Generación automática de contenidos 3D con NeRF

**Alumna:** Vanessa Valentina Villalba Pérez

**Correo:** alu0101265704@ull.edu.es

Universidad de La Laguna

Escuela Superior de Ingeniería y Tecnología

Grado en Ingeniería Informática

> **Note**
> Este repositorio solamente contiene scripts utilizados y necesarios para el avance de este TFG.

---

## Preparación del entorno
> La versión de Python utilizada es `Python 3.8.10`

Desde la raíz del proyecto ejecutar los siguientes comandos:

* Crear entorno virtual 
```
$ python3 -m venv venv/
```
* Activar entorno virtual
```
$ source venv/bin/activate
```
* Instalar dependencias necesarias
```
$ pip3 install -r requirements.txt
```

## Ejecución del servidor
```
$ python3 server.py
```

## Conexión cliente-servidor
Solamente será necesario ejecutar el servidor y seguidamente ejecutar el cliente (la escena de Unity), donde se especificarán las coordenadas esféricas deseadas y se enviarán para obtener la inferencia de Tiny NeRF y así sucesivamente.

