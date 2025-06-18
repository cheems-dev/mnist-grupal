# Red de Kohonen 3D para Clasificación MNIST

Este proyecto implementa una **Red de Mapas Auto-Organizados (SOM) tridimensional** para la clasificación y visualización del dataset MNIST utilizando C++ y OpenGL.

## ▶️ Demo 

https://github.com/user-attachments/assets/799d73e7-db60-43ab-928d-f55f9123fa0f

## 🎯 Descripción del Proyecto

La aplicación entrena una red neuronal de Kohonen en formato de cubo 3D (8×8×8 neuronas) para procesar dígitos manuscritos del dataset MNIST. La implementación incluye:

- **Entrenamiento**: Red SOM 3D con aprendizaje competitivo
- **Visualización**: Renderizado 3D en tiempo real con OpenGL
- **Persistencia**: Guardado y carga automática de pesos entrenados
- **Evaluación**: Métricas de rendimiento y análisis de activaciones

## 🏗️ Arquitectura del Sistema

### Componentes Principales

1. **`Kohonen3D`**: Estructura de la red neuronal
   - Cubo de 8×8×8 = 512 neuronas
   - Cada neurona tiene 784 pesos (28×28 píxeles)
   - Almacenamiento en estructura 4D: `weights[x][y][z][pixel]`

2. **`SOMVisualizer`**: Motor de visualización OpenGL
   - Shaders GLSL para renderizado
   - Cámara orbital automática
   - Texturizado dinámico de patrones neuronales

### Algoritmo de Entrenamiento

```cpp
// Parámetros de configuración
const int SOM_SIZE = 8;        // Tamaño del cubo
const int INPUT_SIZE = 784;    // 28×28 píxeles
const int EPOCHS = 100;        // Épocas de entrenamiento
const int SAMPLES = 1000;      // Muestras por época
```

**Proceso de entrenamiento:**
1. **Inicialización**: Pesos aleatorios [0,1]
2. **BMU (Best Matching Unit)**: Encuentra la neurona más similar usando distancia euclidiana
3. **Actualización**: Modifica pesos con función gaussiana
4. **Decaimiento**: Reduce tasa de aprendizaje y radio de influencia

## 🎮 Funcionalidades

### Entrenamiento Automático
- Detecta pesos preentrenados en `resultados/som_weights.bin`
- Si no existen, entrena automáticamente la red
- Progreso detallado con métricas en tiempo real:
  ```
  Época  95/100 | Progreso: 95.0% | Tasa: 1.000e-02 | Radio: 0.50 | Dist: 0.1234
  ```

### Visualización 3D Interactiva
- **Rotación automática**: Cámara orbital continua
- **Renderizado selectivo**: Solo muestra neuronas de superficie (optimización)
- **Patrones dinámicos**: Cada neurona muestra su patrón aprendido como textura
- **Control**: Presiona `ESC` para salir

### Evaluación de Rendimiento
- Análisis con muestras de prueba
- Métricas de activación neuronal
- Umbral de precisión configurable

## 🛠️ Compilación y Ejecución

### Dependencias
```bash
# Ubuntu/Debian
sudo apt install libglfw3-dev libglew-dev libglm-dev

# Archivos MNIST requeridos:
# - t10k-images-idx3-ubyte
# - t10k-labels-idx1-ubyte  
# - train-images-idx3-ubyte
# - train-labels-idx1-ubyte
```

### Construcción
```bash
mkdir build && cd build
cmake ..
make
./mnist_som_3d
```

## 🎬 Resultado Visual: cubo-3d.mp4

El archivo `cubo-3d.mp4` contiene la **grabación de la visualización 3D** del SOM entrenado. En el video se puede observar:

### Características del Video
- **Duración**: Demostración completa de la red entrenada
- **Visualización**: Cubo 3D rotando automáticamente
- **Contenido**: Patrones neuronales organizados espacialmente
- **Calidad**: Renderizado OpenGL en tiempo real

### Interpretación Visual
- **Neuronas activas**: Muestran patrones similares a dígitos MNIST
- **Organización espacial**: Neuronas vecinas aprenden patrones similares
- **Superficie del cubo**: Solo se renderizan las 248 neuronas exteriores (optimización)
- **Texturas dinámicas**: Cada cara muestra el patrón específico de esa neurona

El video demuestra cómo la red ha **auto-organizado** los dígitos del 0-9 en el espacio tridimensional, donde neuronas cercanas responden a patrones similares.

## 🔧 Configuración Avanzada

### Parámetros Ajustables (`main.cpp:18-26`)
```cpp
const int SOM_SIZE = 8;        // Tamaño del cubo (8³ = 512 neuronas)
const int EPOCHS = 100;        // Épocas de entrenamiento
const int SAMPLES = 1000;      // Muestras por época
const int TEST_SAMPLES = 200;  // Muestras para evaluación
```

### Archivos Generados
- `resultados/som_weights.bin`: Pesos entrenados (persistencia)
- `cubo-3d.mp4`: Video de la visualización 3D

## 📊 Métricas y Rendimiento

### Estadísticas del Modelo
- **Neuronas totales**: 512 (8×8×8)
- **Neuronas de superficie**: 248 (optimización visual)
- **Dimensiones entrada**: 784 (28×28 píxeles)
- **Tiempo entrenamiento**: ~2-5 minutos (depende del hardware)

### Evaluación
La evaluación utiliza un sistema simplificado que:
1. Encuentra la BMU para cada muestra de prueba
2. Calcula métricas de activación neuronal
3. Reporta precisión basada en umbral de distancia

## 🎯 Aplicaciones

Este proyecto demuestra:
- **Reducción dimensional**: De 784D a coordenadas 3D
- **Clustering no supervisado**: Agrupación automática de patrones
- **Visualización científica**: Representación 3D de datos alta dimensional
- **Procesamiento en tiempo real**: Renderizado interactivo OpenGL

## 🔬 Detalles Técnicos

### Algoritmo SOM
- **Función de distancia**: Euclidiana L2
- **Función de vecindad**: Gaussiana 3D
- **Decaimiento**: Exponencial para tasa de aprendizaje y radio
- **Inicialización**: Uniforme aleatoria [0,1]

### OpenGL Pipeline
- **Shaders**: Vertex + Fragment GLSL 3.3
- **Geometría**: Planos cuadrados por neurona
- **Texturizado**: Conversión dinámica pesos → imagen 28×28
- **Proyección**: Perspectiva con FOV 45°

---

*Proyecto desarrollado en C++ con OpenGL para demostración de Redes de Kohonen 3D aplicadas a MNIST.*
