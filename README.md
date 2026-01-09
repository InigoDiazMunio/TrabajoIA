# TrabajoIA
Este repositorio es para llevar acabo el proyecto sobre Entrenamiento distribuido de modelos de inteligencia artificial en entornos multi-GPU de la asignatura Trabajo integrado de computación e IA  de la Universidad de Deusto. Cualquier uso de este codigo ha de ser consultado con el creador.

# NOTA IMPORTANTE
El trabajo, se ha desarrollado en Colab en tres notebooks distintos ( están disponibles en la carpeta notebooks), que es de donde
se han sacado los resultados. El codigo que aparece fuera de los notebooks, puede que no funcione, ya que no simula multiples GPUs. 
##  Descripción

Este proyecto implementa y compara diferentes estrategias de paralelización para el entrenamiento de modelos de deep learning:

- **Baseline**: Entrenamiento en single GPU sin paralelización
- **DataParallel**: Paralelización a nivel de datos 
- **DistributedDataParallel (DDP)**: Entrenamiento distribuido multi-GPU 

## Instalación
```bash
# Clonar repositorio
git clone <repo-url>
cd TRABAJOIA

# Instalar dependencias
pip install -r requirements.txt
```

##  Uso

### Ejecutar todos los experimentos:
```bash
python src/experiments/run_all_experiments.py
```

### Usar módulos individuales:
```python
from src.datasets import get_mnist
from src.models import MLP
from src.training import train_model

# Cargar datos
train_loader, test_loader = get_mnist(batch_size=64)

# Crear modelo
model = MLP()

# Entrenar
device = 'cuda' if torch.cuda.is_available() else 'cpu'
resultados = train_model(
    model, 
    train_loader, 
    test_loader, 
    epochs=5, 
    device=device, 
    name="Mi Experimento"
)
```

##  Experimentos Incluidos

1. **MNIST + MLP**: Clasificación básica
2. **MNIST + CNN Small**: CNN pequeña
3. **CIFAR-10 + CNN Medium**: CNN mediana
4. **CIFAR-100 + CNN Large**: ResNet con bloques residuales
5. **IMDB + Transformer Small**: Clasificación de sentimientos
6. **WikiText-103 + Transformer Medium**: Language modeling

##  Autor

Iñigo Diaz Munio
