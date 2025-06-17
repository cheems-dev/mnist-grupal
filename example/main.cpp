#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <filesystem>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "mnist/mnist_reader.hpp"

namespace fs = std::filesystem;

// Par�metros configurables
const int SOM_SIZE = 8;       // Tama�o del cubo SOM (SOM_SIZE^3 neuronas)
const int INPUT_SIZE = 784;   // 28x28 p�xeles
const int EPOCHS = 100;       // Reducido para pruebas
const int SAMPLES = 1000;     // Subconjunto de entrenamiento
const int TEST_SAMPLES = 200; // Muestras para evaluaci�n

// Directorio de resultados
const std::string RESULT_DIR = "resultados";

// Estructura para la red Kohonen 3D
struct Kohonen3D {
    std::vector<std::vector<std::vector<std::vector<float>>>> weights;
    
    Kohonen3D() {
        weights.resize(SOM_SIZE, 
            std::vector<std::vector<std::vector<float>>>(
                SOM_SIZE,
                std::vector<std::vector<float>>(
                    SOM_SIZE,
                    std::vector<float>(INPUT_SIZE)
                )
            ));
    }
};

// Clase para visualizaci�n OpenGL
class SOMVisualizer {
private:
    GLFWwindow* window;
    Kohonen3D som;
    GLuint shaderProgram;
    GLuint VAO, VBO;
    glm::mat4 projection;
    float rotationAngle;
    int totalNeurons;
    int surfaceNeurons;
    bool trainingCompleted;
    bool weightsLoaded;
    
public:
    SOMVisualizer() : window(nullptr), rotationAngle(0.0f), 
                      totalNeurons(0), surfaceNeurons(0),
                      trainingCompleted(false), weightsLoaded(false) {}
    
    bool initGL() {
        // Inicializar GLFW
        if (!glfwInit()) return false;
        
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        
        window = glfwCreateWindow(1200, 900, "SOM 3D - MNIST", NULL, NULL);
        if (!window) {
            glfwTerminate();
            return false;
        }
        
        glfwMakeContextCurrent(window);
        glewExperimental = GL_TRUE;
        if (glewInit() != GLEW_OK) return false;
        
        setupShaders();
        setupBuffers();
        glEnable(GL_DEPTH_TEST);
        
        // Configurar proyecci�n
        projection = glm::perspective(
            glm::radians(45.0f), 
            1200.0f / 900.0f, 
            0.1f, 
            100.0f
        );
        
        return true;
    }










//jayan - run y main
void run(mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t>& dataset) {
        initializeSOM();
        
        if (!weightsLoaded) {
            trainSOM(dataset);
        }
        else {
            trainingCompleted = true;
        }
        
        // Evaluar rendimiento
        evaluatePerformance(dataset);
        
        // Inicializar y mostrar OpenGL
        if (!initGL()) {
            std::cerr << "Error al inicializar OpenGL" << std::endl;
            return;
        }
        
        // Bucle principal de renderizado
        while (!glfwWindowShouldClose(window)) {
            rotationAngle += 0.005f; // Rotación automática
            
            renderSurface();
            glfwSwapBuffers(window);
            glfwPollEvents();
            
            // Salir con ESC
            if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
                glfwSetWindowShouldClose(window, true);
            }
        }
        
        // Limpieza
        glDeleteVertexArrays(1, &VAO);
        glDeleteBuffers(1, &VBO);
        glfwTerminate();
    }
};

int main() {
    // Cargar dataset MNIST
    std::cout << "Cargando dataset MNIST..." << std::endl;
    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
        mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);
    
    std::cout << "Datos cargados:" << std::endl;
    std::cout << " - Muestras entrenamiento: " << dataset.training_images.size() << std::endl;
    std::cout << " - Muestras prueba: " << dataset.test_images.size() << std::endl;
    
    // Crear y ejecutar visualizador
    SOMVisualizer visualizer;
    visualizer.run(dataset);
    
    return 0;
}
