from itertools import groupby, combinations
from numpy.typing import NDArray
import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import statistics
import os
import time
import datetime

# Individuo, son dos matrices, una que tiene los números y otro que indica las casillas ocupadas además de su aptitud
class Individuo:
    """Es un elemento de la población, es decir, una solución probable. Computesta por dos matrices, una que tiene los números y otro que indica las casillas ocupadas además de su aptitud."""
    def __init__(self, matriz_numeros: NDArray, matriz_posiciones: NDArray):
        self._matriz_numeros: NDArray = matriz_numeros
        self._matriz_posiciones: NDArray = matriz_posiciones
        self._aptitud: int = None
        self._asignar_aptitud()

    @staticmethod
    def _no_satisface_regla(arr: NDArray) -> bool:
        """Suma todos los valores que son diferentes"""
        # Si el array es una matriz, lo aplana para evitar problemas de comparación
        if len(arr.shape) > 1:
            arr = arr.flatten()
        # Sumar valores repetidos
        conteo = np.bincount(arr.astype(int))

        return np.sum(conteo[conteo > 1]) 
    
    def _submatrices(self) -> List[NDArray]:
        """Devuelve una lista de submatrices de 3x3."""
        matriz = self.obtener_matriz_numeros()
        filas, columnas = matriz.shape
        submatrices = []
        # Obtener submatrices de 3x3
        salto = 3
        for i in range(0, filas, salto):
            for j in range(0, columnas, salto):
                submatriz = matriz[i:i+salto, j:j+salto]  # Extraer la submatriz de 3x3
                submatrices.append(submatriz)
        return submatrices

    def _asignar_aptitud(self):
        # Verificar si la matriz satisface la regla en las columnas
        errores_columnas = np.apply_along_axis(self._no_satisface_regla, 0, self.obtener_matriz_numeros())
        # Verificar si la matriz satisface la regla en las submatrices
        submatrices = self._submatrices()
        errores_submatrices = [self._no_satisface_regla(submatriz) for submatriz in submatrices]
        # Sumar los errores de las columnas y submatrices
        self._aptitud = sum(errores_columnas) + sum(errores_submatrices)
    
    def obtener_matriz_numeros(self) -> NDArray:
        return self._matriz_numeros
    
    def obtener_matriz_posiciones(self) -> NDArray:
        return self._matriz_posiciones
    
    def obtener_aptitud(self) -> int:
        return self._aptitud

    def __lt__(self, individuo):  # Menor que
        """Compara dos individuos, si el primero tiene aptitud menor que el segundo devuelve True."""
        return self.obtener_aptitud() < individuo.obtener_aptitud()

    def __le__(self, individuo):  # Menor o igual que
        """Compara dos individuos, si el primero tiene aptitud menor o igual que el segundo devuelve True."""
        return self.obtener_aptitud() <= individuo.obtener_aptitud()

    def __gt__(self, individuo):  # Mayor que
        """Compara dos individuos, si el primero tiene aptitud mayor que el segundo devuelve True."""
        return self.obtener_aptitud() > individuo.obtener_aptitud()

    def __ge__(self, individuo):  # Mayor o igual que
        """Compara dos individuos, si el primero tiene aptitud mayor o igual que el segundo devuelve True."""
        return self.obtener_aptitud() >= individuo.obtener_aptitud()

    def __eq__(self, individuo): # Igual que
        """Compara dos individuos, si son iguales devuelve True."""
        return np.array_equal(self._matriz_numeros, individuo.obtener_matriz_numeros()) 

    def __ne__(self, individuo):  # Diferente
        """Compara dos individuos, si son diferentes devuelve True."""
        return not self.__eq__(individuo)  # Si son diferentes devuelve True


class Cruza:
    """Clase para realizar la cruza entre individuos."""
    def __init__(self, pc1: float = 0.2, pc2: float = 0.1):
        self.pc1 = pc1  # Probabilidad de cruza individual
        self.pc2 = pc2  # Probabilidad de cruza de fila
    
    def cruzar_parejas(self, parejas: List[Tuple[Individuo, Individuo]]) -> List[Individuo]:
        """Realiza la cruza entre parejas de individuos para generar nuevos hijos."""
        hijos = []
        
        for padre1, padre2 in parejas:
            # Crear copias de los padres para no modificar los originales
            matriz1 = padre1.obtener_matriz_numeros().copy()
            matriz2 = padre2.obtener_matriz_numeros().copy()
            matriz_pos = padre1.obtener_matriz_posiciones().copy()
            
            # Decidir si se cruzan basado en pc1
            if np.random.random() < self.pc1:
                # Seleccionar filas para intercambiar basado en pc2
                for i in range(matriz1.shape[0]):
                    if np.random.random() < self.pc2:
                        # Intercambiar filas
                        temp = matriz1[i].copy()
                        matriz1[i] = matriz2[i].copy()
                        matriz2[i] = temp
            
            # Crear nuevos individuos con las matrices cruzadas
            hijo1 = Individuo(matriz1, matriz_pos)
            hijo2 = Individuo(matriz2, matriz_pos)
            
            hijos.extend([hijo1, hijo2])
        
        return hijos


class Mutador:
    """Clase para realizar mutaciones en los individuos."""
    def __init__(self, pm1: float = 0.3, pm2: float = 0.05):
        self.pm1 = pm1  # Probabilidad de mutación por intercambio
        self.pm2 = pm2  # Probabilidad de mutación por reinicialización
    
    def mutar_poblacion(self, poblacion: List[Individuo]) -> List[Individuo]:
        """Realiza la mutación de la población."""
        poblacion_mutada = []
        
        for individuo in poblacion:
            matriz_numeros = individuo.obtener_matriz_numeros().copy()
            matriz_posiciones = individuo.obtener_matriz_posiciones().copy()
            
            # Aplicar mutación por intercambio
            filas = matriz_numeros.shape[0]
            for fila in range(filas):
                if np.random.random() < self.pm1:
                    # Obtener posiciones no asignadas (donde matriz_posiciones[fila] == 0)
                    posiciones_no_asignadas = np.where(matriz_posiciones[fila] == 0)[0]
                    
                    # Si hay al menos 2 posiciones no asignadas, realizar intercambio
                    if len(posiciones_no_asignadas) >= 2:
                        p1, p2 = np.random.choice(posiciones_no_asignadas, size=2, replace=False)
                        # Intercambiar valores
                        temp = matriz_numeros[fila, p1]
                        matriz_numeros[fila, p1] = matriz_numeros[fila, p2]
                        matriz_numeros[fila, p2] = temp
            
            # Aplicar mutación por reinicialización
            for fila in range(matriz_numeros.shape[0]):
                if np.random.random() < self.pm2:
                    # Obtener valores asignados y no asignados
                    posiciones_no_asignadas = np.where(matriz_posiciones[fila] == 0)[0]
                    valores_asignados = [matriz_numeros[fila, columna] for columna in range(matriz_numeros.shape[1]) 
                                        if matriz_posiciones[fila, columna] == 1]
                    valores_disponibles = [v for v in range(1, matriz_numeros.shape[1] + 1) 
                                          if v not in valores_asignados]
                    
                    # Mezclar valores disponibles
                    np.random.shuffle(valores_disponibles)
                    
                    # Reasignar valores a posiciones no asignadas
                    for i, pos in enumerate(posiciones_no_asignadas):
                        matriz_numeros[fila, pos] = valores_disponibles[i]
            
            # Crear nuevo individuo mutado
            individuo_mutado = Individuo(matriz_numeros, matriz_posiciones)
            poblacion_mutada.append(individuo_mutado)
        
        return poblacion_mutada

    def obtener_porcentaje_mutacion_reinicializacion(self) -> float:
        """Devuelve el porcentaje de mutación por reinicialización."""
        return self.pm2
    
    def actualizar_porcentaje_mutacion_reinicializacion(self, nuevo_porcentaje: float):
        """Actualiza el porcentaje de mutación por reinicialización."""
        self.pm2 = nuevo_porcentaje
    
    def reducir_porcentaje_mutacion_reinicializacion(self, porcentaje: float):
        """Reduce el porcentaje de mutación por reinicialización."""
        self.pm2 -= porcentaje
        if self.pm2 < 0:
            self.pm2 = 0.0  # Asegurarse de que no sea negativo
    
    def aumentar_porcentaje_mutacion_reinicializacion(self, porcentaje: float):
        """Aumenta el porcentaje de mutación por reinicialización."""
        self.pm2 += porcentaje
        if self.pm2 > 1:
            self.pm2 = 1.0  # Asegurarse de que no supere 1.0 (100%)

class BusquedaLocal:
    """Clase para realizar búsqueda local en columnas y sub-bloques."""
    def __init__(self, habilitar_busqueda: bool = True):
        self.habilitar_busqueda = habilitar_busqueda

    @staticmethod
    def pares_columnas(columnas_ilegales: List[int]) -> List[Tuple[int, int]]:
        """Genera pares de columnas ilegales de manera aleatoria."""
        # Barajar las columnas ilegales
        np.random.shuffle(columnas_ilegales)
        
        # Creamos todas las combinaciones posibles
        pares = list(combinations(columnas_ilegales, 2))

        return pares
    
    @staticmethod
    def _numeros_repetidos_columna(matriz: NDArray, fila: int, col1: int, col2: int) -> bool:
        """Verifica si los números en las columnas son repetidos en su columna, si lo son, verifica si se pueden intercambiar
        observando si ya existe el número a intercambiar en la otra columna, si no están, entonces se puede realizar"""
            
        num_1 = matriz[fila, col1]
        num_2 = matriz[fila, col2]
        
        # Verificar si el número a intercambiar está repetido en la columna
        repetido_1 = np.sum(matriz[:, col1] == num_1) > 1
        repetido_2 = np.sum(matriz[:, col2] == num_2) > 1

        if repetido_1 and repetido_2:
            # Verificar si ya existe el número a intercambiar en la otra columna
            if np.sum(matriz[:, col1] == num_2) > 0 and np.sum(matriz[:, col2] == num_1) > 0:
                return False
            else:
                # Sólo se puede realizar el cambio si los números a intercambiar no están en la otra columna
                return True
        else:
            return False
        
    
    def _busqueda_columna(self, individuo: Individuo) -> Individuo:
        """Realiza búsqueda local en las columnas del individuo."""
        if not self.habilitar_busqueda:
            return individuo
            
        matriz = individuo.obtener_matriz_numeros().copy()
        matriz_pos = individuo.obtener_matriz_posiciones().copy()
        num_colums = matriz.shape[1]

        # Identificar columnas ilegales (aquellas que no cumplen la regla de unicidad)
        columnas_ilegales = []
        for col in range(num_colums):
            columna = matriz[:, col]
            if len(columna) != len(set(columna)):
                columnas_ilegales.append(col)

        pares = self.pares_columnas(columnas_ilegales)
        # Si hay al menos 2 columnas ilegales, intentar arreglarlas
        for par in pares:        
            col1, col2 = par
            # Buscar valores repetidos en ambas columnas
            for fila in range(matriz.shape[0]):
                # Si ambas posiciones no son fijas (pueden modificarse)
                if matriz_pos[fila, col1] == 0 and matriz_pos[fila, col2] == 0:
                    if self._numeros_repetidos_columna(matriz, fila, col1, col2):
                        # Realizar intercambio tentativo
                        temp = matriz.copy()
                        temp[fila, col1], temp[fila, col2] = temp[fila, col2], temp[fila, col1]
                        
                        # Evaluar mejora
                        individuo_actual = Individuo(matriz, matriz_pos)
                        individuo_nuevo = Individuo(temp, matriz_pos)
                        
                        if individuo_nuevo < individuo_actual:
                            # El intercambio mejora la aptitud
                            matriz = temp.copy()
        
        return Individuo(matriz, matriz_pos)
    
    @staticmethod
    def parejas(lista: List[tuple]) -> List[List[tuple]]:
        # Obtener las sublistas de los puntos que tienen la misma coordenada x y tienen más de 2 elementos
        sublistas = [list(g) for _, g in groupby(lista, key=lambda x: x[0])]
        # Eliminar las sublistas que tienen menos de 2 elementos
        sublistas = [sublista for sublista in sublistas if len(sublista) >= 2]
        # Desordenar los elementos de las sublistas
        for sublista in sublistas:
            np.random.shuffle(sublista)  # Barajar los elementos de la sublista

        parejas = []  # Lista para almacenar lass parejas
        # Agrupar de dos en dos cada elemento de las sublistas con otro random
        for grupo in sublistas:
            # Como ya están desordenados y no pueden ser mayores a 3
            # Si el grupo es de 2, sólo hay una posibilidad
            if len(grupo) == 2:
                parejas.append((grupo[0], grupo[1]))
            else:
                # Si es de 3, ahora cada elemento puede emparejarse con uno de los dos
                for i, elemento in enumerate(grupo):
                    siguiente = (i + 1) % len(grupo)  # El siguiente elemento en el grupo
                    parejas.append((elemento, grupo[siguiente]))
        return parejas

    @staticmethod
    def _numeros_repetidos_subloque(matriz: NDArray, p1: tuple, p2: tuple, salto: int, posbloq1: tuple, psobloq2: tuple) -> bool:
        """Verifica si los números están repetidos en el subloque, si lo son, verifica si se pueden intercambiar observando
        si ya existe el número a intercambiar en el otro subloque, si no están, entonces se puede realizar
        el cambio."""
        
        num_1 = matriz[p1]
        num_2 = matriz[p2]
        submatriz_1 = matriz[posbloq1[0]:posbloq1[0]+salto, posbloq1[1]:posbloq1[1]+salto]
        submatriz_2 = matriz[psobloq2[0]:psobloq2[0]+salto, psobloq2[1]:psobloq2[1]+salto]

        # Verificar si el número a intercambiar está repetido en el respectivo subbloque
        repetido_1 = np.sum(submatriz_1 == num_1) > 1
        repetido_2 = np.sum(submatriz_2 == num_2) > 1

        if repetido_1 and repetido_2:
            # Verificar si ya existe el número a intercambiar en el otro subbloque
            if np.sum(submatriz_1 == num_2) > 0 and np.sum(submatriz_2 == num_1) > 0:
                return False
            else:
                # Sólo se puede realizar el cambio si los números a intercambiar no están en el otro subbloque
                return True
        else:
            return False


    def _busqueda_subbloque(self, individuo: Individuo) -> Individuo:
        """Realiza búsqueda local en los sub-bloques del individuo."""
        if not self.habilitar_busqueda:
            return individuo
            
        matriz = individuo.obtener_matriz_numeros().copy()
        matriz_pos = individuo.obtener_matriz_posiciones().copy()
        filas = matriz.shape[0]
        # salto = int(np.sqrt(filas))
        assert filas == 9, "El Sudoku debe ser de 9x9"
        salto = 3  # Asumiendo que el Sudoku es de 9x9, los sub-bloques son de 3x3

        # Identificar sub-bloques ilegales
        subbloques_ilegales = []
        for i in range(0, filas, salto):
            for j in range(0, filas, salto):
                subbloque = matriz[i:i+salto, j:j+salto]
                # Verificar si el sub-bloque tiene valores repetidos
                if len(np.unique(subbloque)) != filas:
                    subbloques_ilegales.append((i, j))
        
        # Agrupar los bloques ilegales en pares
        pares = self.parejas(subbloques_ilegales)

        for par in pares:
            sb1_i, sb1_j = par[0]
            sb2_i, sb2_j = par[1]
            
            # Buscar filas con valores repetidos en ambos sub-bloques
            for fila_offset in range(salto):
                fila1 = sb1_i + fila_offset
                fila2 = sb2_i + fila_offset
                
                # Verificar si hay posiciones intercambiables en esta fila
                posiciones1 = [(fila1, sb1_j + j) for j in range(salto) if matriz_pos[fila1, sb1_j + j] == 0]
                posiciones2 = [(fila2, sb2_j + j) for j in range(salto) if matriz_pos[fila2, sb2_j + j] == 0]

                for p1 in posiciones1:
                    for p2 in posiciones2:
                        if self._numeros_repetidos_subloque(matriz, p1, p2, salto, par[0], par[1]):
                            # Realizar intercambio tentativo
                            temp = matriz.copy()
                            temp[p1], temp[p2] = temp[p2], temp[p1]
                            
                            # Evaluar mejora
                            individuo_actual = Individuo(matriz, matriz_pos)
                            individuo_nuevo = Individuo(temp, matriz_pos)
                            
                            if individuo_nuevo < individuo_actual:
                                # El intercambio mejora la aptitud
                                matriz = temp.copy()
        
        return Individuo(matriz, matriz_pos)
    
    def aplicar_busqueda_local(self, individuo: Individuo) -> Individuo:
        """Aplica búsqueda local en columnas y sub-bloques."""
        if not self.habilitar_busqueda:
            return individuo
            
        # Aplicar búsqueda en columnas
        individuo_1 = self._busqueda_columna(individuo)
        # Aplicar búsqueda en sub-bloques
        individuo_2 = self._busqueda_subbloque(individuo)

        individuo = min(individuo_1, individuo_2)  # Elegir el mejor
        return individuo
    
    def habilitar(self, estado: bool):
        """Habilita o deshabilita la búsqueda local."""
        self.habilitar_busqueda = estado


class TorneoBinario:
    """Clase para seleccionar individuos mediante torneo binario."""
    def __init__(self, poblacion: List[Individuo], tam_torneo: int = 2, estocastico: bool = False):
        self._poblacion = poblacion.copy()
        self._tam_torneo = tam_torneo
        self._estocastico = estocastico
    
    def _barajeo(self):
        """Barajea la población."""
        indices = np.arange(len(self._poblacion))
        np.random.shuffle(indices)
        self._poblacion = [self._poblacion[i] for i in indices]
    
    def _torneo(self) -> List[Individuo]:
        """Realiza un torneo binario, seleccionando dos individuos al azar y eligiendo el mejor."""
        # Barajea la población para seleccionar aleatoriamente
        self._barajeo()
        seleccionados = []
        
        # Seleccionar los mejores en cada torneo
        for i in range(0, len(self._poblacion), self._tam_torneo):
            # Asegurarse de que hay suficientes individuos para el torneo
            if i + self._tam_torneo <= len(self._poblacion):
                # Seleccionar participantes del torneo
                participantes = self._poblacion[i:i+self._tam_torneo]
                
                # Encontrar el mejor participante
                mejor = min(participantes)  # menor aptitud = mejor
                
                # Selección estocástica si está habilitada
                if self._estocastico:
                    # Con probabilidad p, seleccionar el mejor; con 1-p, seleccionar al azar
                    if np.random.random() < 0.8:  # 80% de probabilidad de seleccionar el mejor
                        seleccionados.append(mejor)
                    else:
                        seleccionados.append(np.random.choice(participantes))
                else:
                    # Selección determinista
                    seleccionados.append(mejor)
        
        return seleccionados
    
    def seleccionar_parejas(self) -> List[Tuple[Individuo, Individuo]]:
        """Selecciona parejas de individuos para cruza."""
        # Obtener las listas de padres
        padres_1 = self._torneo()
        padres_2 = self._torneo()

        # Barajear los padres para formar parejas aleatorias
        np.random.shuffle(padres_1)

        assert len(padres_1) == len(padres_2), "Las listas de padres deben tener la misma longitud"

        # Formar parejas
        parejas = []
        for prospecto_1 in padres_1:
            for prospecto_2 in padres_2:
                if prospecto_1 != prospecto_2:
                    parejas.append((prospecto_1, prospecto_2))
                    padres_2.remove(prospecto_2)  # Eliminar el prospecto de la segunda lista para evitar duplicados
                    break # Salir del bucle una vez que se ha encontrado un prospecto diferente
        
        return parejas



class Poblacion:
    """Guarda la población, que contiene a los individuos, a los hijos y padres."""
    def __init__(self, mutador: Mutador, busqueda: BusquedaLocal, cruza: Cruza, 
                 tam_poblacion: int, solucion_parcial: NDArray, tam_elite: int = 50, 
                 dimension: int = 9):
        self.dimension = dimension
        self._poblacion: List[Individuo] = []
        self._mutador = mutador
        self._busqueda = busqueda
        self._cruza = cruza
        self._solucion_parcial = solucion_parcial
        self._tam_pop = tam_poblacion
        self._tam_elite = tam_elite
        self._valores_posibles = np.array([i for i in range(1, 10)])
        self._matriz_posiciones = self._crear_matriz_posicion()
        self._elite = deque(maxlen=tam_elite)  # Cola para mantener la población elite
        self._mejor_individuo = None
        self._porcentaje_mutacion_reinicializacion = mutador.obtener_porcentaje_mutacion_reinicializacion()  # Porcentaje de individuos a reinicializar
        
        # Estadísticas para gráficas
        self.estadisticas = {
            'mejor_aptitud': [],
            'peor_aptitud': [],
            'aptitud_media': [],
            'desviacion_estandar': []
        }

    def _no_cero(self, lista: NDArray) -> NDArray:
        """Devuelve una lista sin los ceros."""
        return lista[lista != 0]
    
    def _no_asignado(self, valores_no_posibles: NDArray) -> NDArray:
        """Devuelve la lista con los números posibles para sustituir"""
        lista_original = self._valores_posibles.copy()
        mask = np.isin(lista_original, valores_no_posibles, invert=True)
        lista_sustitucion = lista_original[mask]
        return lista_sustitucion
    
    def _inicializacion_fila(self, fila: NDArray):
        """Inicializa la fila de un individuo, asignando valores aleatorios de una lista a las posiciones vacías."""
        # Obtener los valores asignados
        asignado = self._no_cero(fila)
        # Obtener los valores no asignados
        sustitutos = self._no_asignado(asignado)
        # Sistituir los valores no asignados por valores aleatorios de la lista de sustitución
        np.random.shuffle(sustitutos)
        mask = fila == 0
        fila[mask] = sustitutos[:np.sum(mask)]
        return fila

    def iniciar_poblacion(self):
        """Inicia la población, creando una lista de individuos."""
        matriz = self._solucion_parcial.copy()
        # Checar que se haya usado una matriz correcta
        assert matriz.shape[0] == self.dimension and matriz.shape[1] == self.dimension, \
              'La matriz no tiene las dimensiones esperadas'

        self._poblacion = []
        for _ in range(self._tam_pop):
            # Cada individuo es una copia de la matriz inicial, pero con los valores no asignados sustituidos por valores aleatorios
            matriz_copia = matriz.copy()
            matriz_individuo = np.apply_along_axis(self._inicializacion_fila, 1, matriz_copia)
            individuo = Individuo(matriz_individuo, self._matriz_posiciones.copy())
            self._poblacion.append(individuo)
        
        # Actualizar el mejor individuo
        self._actualizar_mejor()
        
        # Registrar estadísticas iniciales
        self._actualizar_estadisticas()

    def _crear_matriz_posicion(self) -> NDArray:  
        """Crea una matriz de posiciones, donde 1 indica posición fija y 0 posición variable."""
        matriz = self._solucion_parcial.copy()
        # Crear una matriz de ceros o unos del mismo tamaño que la matriz original
        return np.where(matriz > 0, 1, 0)
    
    def _seleccionar_poblacion(self):
        """Selecciona individuos mediante torneo binario."""
        torneo = TorneoBinario(self._poblacion, tam_torneo=2)
        parejas = torneo.seleccionar_parejas()
        return parejas
    
    def recombinacion(self):
        """Realiza la recombinación de los individuos de la población."""
        # Seleccionar parejas para cruza
        parejas = self._seleccionar_poblacion()
        # Realizar cruza
        hijos = self._cruza.cruzar_parejas(parejas)
        # Realizar mutación
        hijos = self._mutador.mutar_poblacion(hijos)
        return hijos
    
    def _actualizar_mejor(self):
        """Actualiza el mejor individuo de la población."""
        if not self._poblacion:
            return
        
        mejor = min(self._poblacion)
        
        if self._mejor_individuo is None or mejor.obtener_aptitud() < self._mejor_individuo.obtener_aptitud():
            self._mejor_individuo = mejor
    
    def _actualizar_estadisticas(self):
        """Actualiza las estadísticas de la población para graficar."""
        if not self._poblacion:
            return
            
        # Convertir aptitudes a tipos nativos de Python (int) para evitar problemas con statistics.stdev
        aptitudes = [int(ind.obtener_aptitud()) for ind in self._poblacion]
        
        self.estadisticas['mejor_aptitud'].append(min(aptitudes))
        self.estadisticas['peor_aptitud'].append(max(aptitudes))
        self.estadisticas['aptitud_media'].append(sum(aptitudes) / len(aptitudes))
        
        # Calcular la desviación estándar solo si tenemos más de 1 individuo
        if len(aptitudes) > 1:
            self.estadisticas['desviacion_estandar'].append(statistics.stdev(aptitudes))
        else:
            self.estadisticas['desviacion_estandar'].append(0)
    
    def _aprendizaje_elite(self):
        """Implementa el mecanismo de aprendizaje de población elite.
        Los mejores se van guardando en la población elite, hasta un máximo definido.
        Cada vez que se reemplaza al peor individuo, se elige uno de la población elite o se reinicializa."""
        # Ingresamos al mejor individuo a la población elite
        mejor_individuo = self._mejor_individuo
        if mejor_individuo not in self._elite:
            self._elite.append(self._poblacion[0])

        # Equilibrio entre exploración y explotación
        if np.random.random() < 0.5:
            # Reemplazar el peor individuo por uno de la élite
            if len(self._elite) > 0:
                # Seleccionar un individuo élite al azar
                elite_seleccionado = np.random.choice(self._elite)
                # Reemplazar el peor individuo por el élite seleccionado
                self._poblacion[-1] = elite_seleccionado
        else:
            # Si no, reinicializar el peor individuo
            matriz_copia = self._solucion_parcial.copy()
            matriz_individuo = np.apply_along_axis(self._inicializacion_fila, 1, matriz_copia)
            self._poblacion[-1] = Individuo(matriz_individuo, self._matriz_posiciones.copy())



    def _aprendizaje_elite_mejorado(self):
        """Implementa un mecanismo mejorado de aprendizaje de población elite con control de diversidad."""
        # Ordenar la población por aptitud (de mejor a peor)
        self._poblacion.sort()
        
        # Lista para candidatos a élite en esta generación
        candidatos_elite = []
        
        # Seleccionar los mejores candidatos
        for i in range(min(10, len(self._poblacion))):  # Considerar un grupo mayor de candidatos
            if self._poblacion[i].obtener_aptitud() == 0:
                # Si ya es una solución óptima, la agregamos directamente a la élite
                self._elite.append(self._poblacion[i])
                continue
                
            candidatos_elite.append(self._poblacion[i])
        
        # Filtrar candidatos por diversidad antes de agregarlos a la élite
        for candidato in candidatos_elite:
            # Verificar si el candidato aporta diversidad a la élite
            es_diverso = True
            
            # Si la élite está vacía, cualquier candidato es válido
            if len(self._elite) == 0:
                self._elite.append(candidato)
                continue
                
            # Calcular similitud con la élite existente
            for elite in self._elite:
                # Calcular qué tan similares son las matrices
                matriz_candidato = candidato.obtener_matriz_numeros()
                matriz_elite = elite.obtener_matriz_numeros()
                
                # Porcentaje de celdas idénticas (medida de similitud)
                similitud = np.sum(matriz_candidato == matriz_elite) / matriz_candidato.size
                
                if similitud > 0.8:  # Si más del 80% de celdas son iguales, considerarlo no diverso
                    es_diverso = False
                    break
                    
            # Solo añadir si es suficientemente diverso o mejor que alguno de la élite actual
            if es_diverso or candidato.obtener_aptitud() < min([e.obtener_aptitud() for e in self._elite], default=float('inf')):
                self._elite.append(candidato)
        
        # Limitar el tamaño de la élite al máximo definido
        if len(self._elite) > self._tam_elite:
            # Ordenar la élite por aptitud y quedarse con los mejores
            self._elite = deque(sorted(self._elite)[:self._tam_elite], maxlen=self._tam_elite)
        
        # Si la población elite está vacía, no realizar reemplazo
        if not self._elite:
            return
        
        # MEJORA: Reemplazar individuos con estrategia adaptativa
        num_reemplazos = int(0.15 * len(self._poblacion))  # Aumentar a 15% para tener más impacto
        
        # Ordenar élite por aptitud (mejor a peor)
        elite_ordenada = sorted(self._elite)
        
        for i in range(num_reemplazos):
            if i < len(self._poblacion):
                # Índice del individuo a potencialmente reemplazar
                idx = len(self._poblacion) - 1 - i  # Comenzar por los peores
                
                # MEJORA: Calcular probabilidad de reemplazo usando función sigmoide
                # para transición más suave entre reemplazar y no reemplazar
                ratio_aptitud = self._poblacion[idx].obtener_aptitud() / (self._poblacion[0].obtener_aptitud() + 1)
                prob_reemplazo = 1 / (1 + np.exp(-10 * (ratio_aptitud - 0.5)))  # Función sigmoide
                
                if np.random.random() < prob_reemplazo:
                    # Seleccionar un individuo élite basado en la diversidad
                    if len(elite_ordenada) > 0:
                        # Con 70% de probabilidad elegir uno de los mejores élites
                        if np.random.random() < 0.7:
                            # Elegir aleatoriamente entre el top 30% de la élite
                            top_index = max(1, int(0.3 * len(elite_ordenada)))
                            elite_seleccionado = np.random.choice(elite_ordenada[:top_index])
                        else:
                            # Elegir completamente al azar de la élite para mantener diversidad
                            elite_seleccionado = np.random.choice(elite_ordenada)
                            
                        self._poblacion[idx] = elite_seleccionado
                    else:
                        # Si no hay élite (aunque debería haberla a este punto), reinicializar
                        matriz_copia = self._solucion_parcial.copy()
                        matriz_individuo = np.apply_along_axis(self._inicializacion_fila, 1, matriz_copia)
                        self._poblacion[idx] = Individuo(matriz_individuo, self._matriz_posiciones.copy())

    def obtener_poblacion_unica(self, individuos):
        """Retorna una lista de individuos sin duplicados."""
        hash_dict = {}
        for ind in individuos:
            # Usar hash del array como identificador
            h = hash(ind.obtener_matriz_numeros().tobytes())
            hash_dict[h] = ind
        
        return list(hash_dict.values())

    def evolucionar(self, max_generaciones: int = 10000, 
                umbral_estancamiento: int = 50):
        """
        Evoluciona la población hasta encontrar una solución o alcanzar el máximo de generaciones.
        
        Args:
            max_generaciones: Número máximo de generaciones
            umbral_estancamiento: Número de generaciones sin mejora para considerar estancamiento
        
        Returns:
            Individuo: El mejor individuo encontrado
        """
        # Inicializar la población
        self.iniciar_poblacion()
        
        generacion = 0
        # Para detectar estancamiento
        mejor_aptitud_anterior = float('inf')
        generaciones_sin_mejora = 0
        max_generaciones_sin_mejora = umbral_estancamiento
        
        while generacion < max_generaciones:
            # Realizar recombinación y mutación
            hijos = self.recombinacion()
            
            # Aplicar búsqueda local a cada hijo
            for i in range(len(hijos)):
                hijos[i] = self._busqueda.aplicar_busqueda_local(hijos[i])
            
            # Combinar población actual con hijos
            poblacion_combinada = self.obtener_poblacion_unica(self._poblacion + hijos)
            
            # Ordenar por aptitud y seleccionar los mejores
            poblacion_combinada.sort()
            self._poblacion = poblacion_combinada[:self._tam_pop]
            
            # Actualizar mejor individuo
            self._actualizar_mejor()
            
            # Aplicar mecanismo de aprendizaje elite mejorado
            self._aprendizaje_elite()
            
            # Actualizar estadísticas
            self._actualizar_estadisticas()
            
            # Verificar si se encontró una solución
            if self._mejor_individuo.obtener_aptitud() == 0:
                print(f"¡Solución encontrada en la generación {generacion}!")
                break
            
            # Detectar mejora
            if self._mejor_individuo.obtener_aptitud() < mejor_aptitud_anterior:
                generaciones_sin_mejora = 0
                mejor_aptitud_anterior = self._mejor_individuo.obtener_aptitud()
            else:
                generaciones_sin_mejora += 1
            
            # Verificar si hay estancamiento
            if generaciones_sin_mejora > max_generaciones_sin_mejora:
                print(f"¡Detectado estancamiento en generación {generacion}. Parando el experimento.!")
                break

            # Mostrar progreso cada 20 generaciones
            if generacion % 20 == 0:
                print(f"+---Generación {generacion}---+:",
                    f"\nMejor aptitud = {self._mejor_individuo.obtener_aptitud()}",
                    f"\nDesviación estándar = {self.estadisticas['desviacion_estandar'][-1]:.3f}",
                    f"\nGeneraciones sin mejora = {generaciones_sin_mejora}",
                    f"\nTotal elite = {len(self._elite)}"
                    )
                    
            # Incrementar contador de generaciones
            generacion += 1
        
        if generacion == max_generaciones:
            print(f"No se encontró una solución óptima después de {max_generaciones} generaciones.")
            print(f"Mejor aptitud alcanzada: {self._mejor_individuo.obtener_aptitud()}")
        
        return self._mejor_individuo

    def obtener_mejor_individuo(self) -> Optional[Individuo]:
        """Devuelve el mejor individuo encontrado."""
        return self._mejor_individuo
    
    def obtener_estadisticas(self) -> Dict[str, List[float]]:
        """Devuelve las estadísticas recopiladas durante la evolución."""
        return self.estadisticas


def mostrar_sudoku(matriz: NDArray):
    """Muestra la matriz de Sudoku de manera formateada."""
    filas = matriz.shape[0]
    sqrt_n = int(np.sqrt(filas))
    
    print("+" + "-" * (2 * filas + sqrt_n - 1) + "+")
    
    for i in range(filas):
        linea = "|"
        for j in range(filas):
            linea += f" {matriz[i, j]}"
            if (j + 1) % sqrt_n == 0 and j < filas - 1:
                linea += " |"
        linea += " |"
        print(linea)
        
        if (i + 1) % sqrt_n == 0 and i < filas - 1:
            print("|" + "-" * (2 * filas + sqrt_n - 1) + "|")
    
    print("+" + "-" * (2 * filas + sqrt_n - 1) + "+")


def graficar_estadisticas(estadisticas: Dict[str, List[float]], guardar_como: str = None, mostrar: bool = True):
    """
    Genera gráficas visualmente atractivas de las estadísticas del algoritmo evolutivo.
    
    Args:
        estadisticas: Diccionario con las estadísticas recopiladas
        guardar_como: Ruta donde guardar la imagen (opcional)
        mostrar: Si se debe mostrar la gráfica en pantalla
    """
    # Configurar el estilo de las gráficas
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Crear una figura con 2 subplots (uno arriba del otro)
    # Usamos constrained_layout en lugar de tight_layout para manejar mejor los elementos
    fig = plt.figure(figsize=(12, 12), dpi=100, constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[10, 10, 1])  # 3 filas, la última para el texto
    
    # Crear los dos subplots principales
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
    # Área para el texto de información adicional
    ax_info = fig.add_subplot(gs[2])
    ax_info.axis('off')  # Ocultar ejes
    
    # Título principal
    fig.suptitle('Evolución del Algoritmo Genético para Sudoku', fontsize=18, fontweight='bold', y=0.98)
    
    # Generar el eje x (generaciones)
    generaciones = list(range(len(estadisticas['mejor_aptitud'])))
    
    # Colores para las líneas con mayor saturación
    mejor_color = '#00A878'  # Verde esmeralda
    peor_color = '#FF6B6B'   # Rojo coral
    media_color = '#118AB2'  # Azul acero
    desv_color = '#8A4FFF'   # Púrpura vibrante (nuevo color para contraste)
    
    # PRIMER SUBPLOT: Aptitudes
    ax1.plot(generaciones, estadisticas['mejor_aptitud'], color=mejor_color, linewidth=2.5, 
                label='Mejor aptitud', marker='o', markevery=max(1, len(generaciones)//20), 
                markersize=6, alpha=0.9)
    ax1.plot(generaciones, estadisticas['peor_aptitud'], color=peor_color, linewidth=2, 
                label='Peor aptitud', marker='^', markevery=max(1, len(generaciones)//20), 
                markersize=6, alpha=0.8)
    ax1.plot(generaciones, estadisticas['aptitud_media'], color=media_color, linewidth=2, 
                label='Aptitud media', marker='s', markevery=max(1, len(generaciones)//20), 
                markersize=6, alpha=0.8)
    
    # Rellenar áreas bajo las curvas para mejor visualización
    ax1.fill_between(generaciones, estadisticas['mejor_aptitud'], estadisticas['aptitud_media'], 
                       color=mejor_color, alpha=0.1)
    ax1.fill_between(generaciones, estadisticas['aptitud_media'], estadisticas['peor_aptitud'], 
                       color=peor_color, alpha=0.1)
    
    # Configurar el primer subplot
    ax1.set_title('Evolución de la Aptitud a lo Largo de las Generaciones', fontsize=15, pad=10)
    ax1.set_xlabel('Generación', fontsize=12)
    ax1.set_ylabel('Aptitud (Errores)', fontsize=12)
    ax1.tick_params(axis='both', which='major', labelsize=10)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Añadir leyenda con mejor ubicación y estilo
    legend = ax1.legend(loc='upper right', fontsize=12, frameon=True, fancybox=True, 
                          shadow=True, facecolor='white', edgecolor='gray')
    
    # Comentario sobre mejor aptitud final
    if len(estadisticas['mejor_aptitud']) > 0:
        mejor_final = estadisticas['mejor_aptitud'][-1]
        ax1.annotate(f'Aptitud final: {mejor_final}', 
                       xy=(len(generaciones)-1, estadisticas['mejor_aptitud'][-1]),
                       xytext=(len(generaciones)-len(generaciones)//4, estadisticas['mejor_aptitud'][-1]+1),
                       arrowprops=dict(facecolor=mejor_color, shrink=0.05, width=1.5, headwidth=8),
                       fontsize=11, fontweight='bold')
    
    # SEGUNDO SUBPLOT: Desviación estándar
    ax2.plot(generaciones, estadisticas['desviacion_estandar'], color=desv_color, linewidth=3, 
                label='Desviación estándar', marker='D', markevery=max(1, len(generaciones)//20), 
                markersize=7)
    
    # Añadir área sombreada bajo la curva con mejor contraste
    ax2.fill_between(generaciones, 0, estadisticas['desviacion_estandar'], 
                       color=desv_color, alpha=0.2,
                       hatch='///', edgecolor=desv_color, linewidth=0)
    
    # Añadir línea de tendencia (media móvil) para mejor visualización
    if len(generaciones) > 5:
        window_size = max(3, len(generaciones) // 30)
        trend = np.convolve(estadisticas['desviacion_estandar'], np.ones(window_size)/window_size, mode='valid')
        trend_x = generaciones[window_size-1:]
        ax2.plot(trend_x, trend, color='#E94560', linewidth=2, linestyle='--', 
                   label='Tendencia', alpha=0.8)
    
    # Configurar el segundo subplot con mejor contraste
    ax2.set_title('Diversidad de la Población (Desviación Estándar)', fontsize=15, pad=10)
    ax2.set_xlabel('Generación', fontsize=12)
    ax2.set_ylabel('Desviación Estándar', fontsize=12)
    ax2.tick_params(axis='both', which='major', labelsize=10)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Configurar el fondo para mejor contraste
    ax2.set_facecolor('#F8F9FA')  # Fondo ligeramente más claro
    
    # Añadir leyenda con mejor ubicación y estilo
    legend = ax2.legend(loc='upper right', fontsize=12, frameon=True, fancybox=True, 
                          shadow=True, facecolor='white', edgecolor='gray')
    
    # Añadir una nota explicativa con mejor diseño
    ax2.annotate('Una desviación estándar menor indica convergencia\nde la población hacia soluciones similares',
                   xy=(len(generaciones)//2, min(estadisticas['desviacion_estandar'])),
                   xytext=(len(generaciones)//4, max(estadisticas['desviacion_estandar'])*0.8),
                   fontsize=10, fontweight='normal', style='italic',
                   bbox=dict(boxstyle="round4,pad=0.5", fc="#FFFFFF", ec=desv_color, alpha=0.9))
    
    # Añadir información adicional en el área de texto inferior
    ax_info.text(0.5, 0.5, f"Total de generaciones: {len(generaciones)}", 
               ha="center", va="center", fontsize=12, fontweight="bold",
               bbox={"facecolor":"white", "edgecolor":"gray", "alpha":0.8, "pad":5})
    
    # Guardar la figura si se proporciona una ruta
    if guardar_como:
        plt.savefig(guardar_como, bbox_inches='tight', dpi=150)
        print(f"Gráfica de estadísticas guardada en: {guardar_como}")
    
    # Mostrar la gráfica si se indica
    if mostrar:
        plt.show()
    else:
        plt.close(fig)


def visualizar_sudoku(matriz: NDArray, matriz_posiciones: NDArray = None, matriz_inicial: NDArray = None, 
                   titulo: str = "Sudoku", guardar_como: str = None):
    """
    Visualiza gráficamente el Sudoku utilizando matplotlib con colores atractivos e informativos.
    
    Parámetros:
    - matriz: La matriz de Sudoku actual a visualizar
    - matriz_posiciones: Matriz que indica cuáles posiciones son fijas (1) y cuáles no (0)
    - matriz_inicial: Matriz del Sudoku inicial para comparación (para identificar celdas corregidas)
    - titulo: Título para el gráfico
    - guardar_como: Ruta donde guardar la imagen (opcional)
    """
    # Verificar dimensiones
    if matriz.shape[0] != 9 or matriz.shape[1] != 9:
        print(f"¡Advertencia! Dimensiones de matriz no estándar: {matriz.shape}")
    
    n = matriz.shape[0]
    sqrt_n = int(np.sqrt(n))
    
    # Crear una figura con layout adecuado para añadir información adicional
    fig = plt.figure(figsize=(10, 11), dpi=100, constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[10, 1])  # 2 filas, la segunda para validez/leyenda
    
    # Área principal para el sudoku
    ax = fig.add_subplot(gs[0])
    
    # Área para la leyenda y validación
    ax_legend = fig.add_subplot(gs[1])
    ax_legend.axis('off')  # Ocultar ejes
    
    # Fondo para toda la figura
    fig.patch.set_facecolor('#F8F9FA')
    
    # Configurar el título con estilo
    ax.set_title(titulo, fontsize=20, fontweight='bold', pad=20, color='#333333')
    
    # Dibujar el grid principal
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_aspect('equal')
    
    # Invertir el eje y para que el (0,0) esté en la esquina superior izquierda
    ax.invert_yaxis()
    
    # Ocultar ejes
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Paleta de colores
    color_fijo = '#E8F4F8'           # Celeste claro para celdas fijas
    color_calculado = '#FFFFFF'      # Blanco para celdas calculadas
    color_calculado_alt = '#F9F9F9'  # Blanco ligeramente oscuro para contraste en bloques
    color_borde_fijo = '#1A85FF'     # Azul para bordes de celdas fijas
    color_borde_bloque = '#333333'   # Negro para bordes principales de bloques
    color_borde_celda = '#BBBBBB'    # Gris para bordes de celdas individuales
    color_texto_fijo = '#0055AA'     # Azul oscuro para números fijos
    color_texto_calculado = '#2E3440' # Gris oscuro para números calculados
    color_texto_corregido = '#00A878' # Verde para números corregidos (si coinciden con la solución)
    color_incorrecto = '#FF5252'     # Rojo para números incorrectos (si hay conflicto)
    
    # Rellenar toda la matriz con un patrón de tablero de ajedrez alternado para los bloques
    for i in range(n):
        for j in range(n):
            # Calcular a qué bloque pertenece esta celda
            bloque_i = i // sqrt_n
            bloque_j = j // sqrt_n
            
            # Decidir color basado en el bloque para crear patrón
            color_base = color_calculado if (bloque_i + bloque_j) % 2 == 0 else color_calculado_alt
            
            # Dibujar el rectángulo de fondo de la celda
            rect = plt.Rectangle((j, i), 1, 1, fill=True, facecolor=color_base,
                               edgecolor=color_borde_celda, linewidth=0.5)
            ax.add_patch(rect)
    
    # Dibujar líneas del grid
    # Líneas finas para todas las celdas
    for i in range(n + 1):
        lw = 3 if i % sqrt_n == 0 else 0.5
        color = color_borde_bloque if i % sqrt_n == 0 else color_borde_celda
        ax.axhline(i, color=color, linewidth=lw)
        ax.axvline(i, color=color, linewidth=lw)
    
    # Verificar si la matriz tiene valores incorrectos (fuera del rango 1-9)
    tiene_valores_incorrectos = False
    for i in range(n):
        for j in range(n):
            if matriz[i, j] < 0 or matriz[i, j] > 9 or (matriz[i, j] != int(matriz[i, j]) and matriz[i, j] != 0):
                tiene_valores_incorrectos = True
                print(f"¡Advertencia! Valor no válido en posición [{i},{j}]: {matriz[i, j]}")
    
    if tiene_valores_incorrectos:
        print("La matriz contiene valores fuera del rango válido (1-9) o no enteros.")
    
    # Colorear celdas según su estado
    for i in range(n):
        for j in range(n):
            # Obtener el valor actual y verificar si es válido
            valor = matriz[i, j]
            if valor != 0 and valor != int(valor):
                valor = int(valor)  # Intentar convertir a entero si es posible
            
            # Verificar si la celda es fija
            es_fija = matriz_posiciones is not None and matriz_posiciones[i, j] == 1
            
            # Verificar si el valor es válido (no tiene conflictos)
            # Comprobamos si hay duplicados en la fila, columna o subcuadro
            es_valido = True
            
            # Comprobar fila
            fila = matriz[i, :]
            if valor != 0 and np.sum(fila == valor) > 1:
                es_valido = False
                
            # Comprobar columna
            columna = matriz[:, j]
            if valor != 0 and np.sum(columna == valor) > 1:
                es_valido = False
                
            # Comprobar subbloque
            bloque_i = (i // sqrt_n) * sqrt_n
            bloque_j = (j // sqrt_n) * sqrt_n
            subbloque = matriz[bloque_i:bloque_i+sqrt_n, bloque_j:bloque_j+sqrt_n]
            if valor != 0 and np.sum(subbloque == valor) > 1:
                es_valido = False
            
            # Verificar si es un valor corregido
            es_corregido = False
            if not es_fija and matriz_inicial is not None and matriz_inicial[i, j] != 0 and matriz_inicial[i, j] != valor:
                es_corregido = True
            
            # Seleccionar color de fondo basado en el estado
            if es_fija:
                # Celdas fijas tienen un color de fondo distinto
                color_fondo = color_fijo
                color_borde = color_borde_fijo
                lw = 2
            else:
                # Color base que ya establecimos en el patrón de ajedrez
                bloque_i = i // sqrt_n
                bloque_j = j // sqrt_n
                color_fondo = color_calculado if (bloque_i + bloque_j) % 2 == 0 else color_calculado_alt
                color_borde = color_borde_celda
                lw = 0.5
                
            # Si es una celda con conflicto, usar color de error
            if not es_valido and valor != 0:
                # Usar un patrón rayado para indicar conflicto
                rect = plt.Rectangle((j, i), 1, 1, fill=True, facecolor=color_incorrecto, alpha=0.2,
                                   edgecolor=color_incorrecto, linewidth=1, hatch='///')
                ax.add_patch(rect)
            else:
                # Dibujar el rectángulo de fondo de la celda
                rect = plt.Rectangle((j, i), 1, 1, fill=True, facecolor=color_fondo,
                                   edgecolor=color_borde, linewidth=lw)
                ax.add_patch(rect)
            
            # Agregar números con colores según su estado
            if valor != 0:
                # Seleccionar color y estilo del texto
                if es_fija:
                    color_texto = color_texto_fijo
                    weight = 'bold'
                elif es_corregido:
                    color_texto = color_texto_corregido
                    weight = 'bold'
                elif not es_valido:
                    color_texto = color_incorrecto
                    weight = 'normal'
                else:
                    color_texto = color_texto_calculado
                    weight = 'normal'
                
                # Dibujar el número
                ax.text(j + 0.5, i + 0.5, str(int(valor)), 
                       va='center', ha='center', fontsize=16, 
                       fontweight=weight, color=color_texto)
    
    # Añadir un borde a toda la figura
    borde_figura = plt.Rectangle((0, 0), n, n, fill=False, edgecolor=color_borde_bloque, linewidth=4)
    ax.add_patch(borde_figura)
    
    # Añadir leyenda en el área separada
    leyenda_texto = ""
    leyenda_texto += "■ Celda fija   "
    leyenda_texto += "■ Celda calculada   "
    
    if matriz_inicial is not None:
        leyenda_texto += "■ Celda corregida   "
    
    leyenda_texto += "■ Conflicto"
    
    ax_legend.text(0.5, 0.7, leyenda_texto, ha='center', va='center', fontsize=12)
    
    # Verificar si la solución es válida
    es_solucion_valida = True
    errores = 0
    
    for i in range(n):
        # Verificar filas
        if len(set(matriz[i, :])) != n or 0 in matriz[i, :]:
            es_solucion_valida = False
            errores += 1
        
        # Verificar columnas
        if len(set(matriz[:, i])) != n or 0 in matriz[:, i]:
            es_solucion_valida = False
            errores += 1
    
    # Verificar sub-bloques
    for i in range(0, n, sqrt_n):
        for j in range(0, n, sqrt_n):
            subbloque = matriz[i:i+sqrt_n, j:j+sqrt_n].flatten()
            if len(set(subbloque)) != n or 0 in subbloque:
                es_solucion_valida = False
                errores += 1
    
    # Añadir indicador de validez
    if "solución" in titulo.lower() or "resuelto" in titulo.lower():
        validez_color = "#00A878" if es_solucion_valida else "#FF5252"
        estado = "VÁLIDA" if es_solucion_valida else f"INVÁLIDA ({errores} errores)"
        ax_legend.text(0.5, 0.2, f"Solución {estado}", 
                   ha="center", va="center", fontsize=14, fontweight="bold", color=validez_color,
                   bbox={"facecolor":"white", "edgecolor":validez_color, "alpha":0.9, "pad":5})
    
    # Guardar imagen si se especifica ruta
    if guardar_como:
        plt.savefig(guardar_como, bbox_inches='tight', dpi=150)
        print(f"Visualización guardada en: {guardar_como}")
    
    # Mostramos el gráfico
    plt.show()

def visualizar_sudoku(matriz: NDArray, matriz_posiciones: NDArray = None, matriz_inicial: NDArray = None, 
                   titulo: str = "Sudoku", guardar_como: str = None):
    """
    Visualiza gráficamente el Sudoku utilizando matplotlib con colores atractivos e informativos.
    
    Parámetros:
    - matriz: La matriz de Sudoku actual a visualizar
    - matriz_posiciones: Matriz que indica cuáles posiciones son fijas (1) y cuáles no (0)
    - matriz_inicial: Matriz del Sudoku inicial para comparación (para identificar celdas corregidas)
    - titulo: Título para el gráfico
    - guardar_como: Ruta donde guardar la imagen (opcional)
    """
    # Verificar dimensiones
    if matriz.shape[0] != 9 or matriz.shape[1] != 9:
        print(f"¡Advertencia! Dimensiones de matriz no estándar: {matriz.shape}")
    
    n = matriz.shape[0]
    sqrt_n = int(np.sqrt(n))
    
    # Crear una figura
    fig, ax = plt.subplots(figsize=(10, 10), dpi=100, facecolor='#F8F9FA')
    
    # Fondo para toda la figura
    plt.gcf().patch.set_facecolor('#F8F9FA')
    
    # Configurar el título con estilo
    ax.set_title(titulo, fontsize=20, fontweight='bold', pad=20, color='#333333')
    
    # Dibujar el grid principal
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_aspect('equal')
    
    # Invertir el eje y para que el (0,0) esté en la esquina superior izquierda
    ax.invert_yaxis()
    
    # Ocultar ejes
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Paleta de colores
    color_fijo = '#E8F4F8'           # Celeste claro para celdas fijas
    color_calculado = '#FFFFFF'      # Blanco para celdas calculadas
    color_calculado_alt = '#F9F9F9'  # Blanco ligeramente oscuro para contraste en bloques
    color_borde_fijo = '#1A85FF'     # Azul para bordes de celdas fijas
    color_borde_bloque = '#333333'   # Negro para bordes principales de bloques
    color_borde_celda = '#BBBBBB'    # Gris para bordes de celdas individuales
    color_texto_fijo = '#0055AA'     # Azul oscuro para números fijos
    color_texto_calculado = '#2E3440' # Gris oscuro para números calculados
    color_texto_corregido = '#00A878' # Verde para números corregidos (si coinciden con la solución)
    color_incorrecto = '#FF5252'     # Rojo para números incorrectos (si hay conflicto)
    
    # Rellenar toda la matriz con un patrón de tablero de ajedrez alternado para los bloques
    for i in range(n):
        for j in range(n):
            # Calcular a qué bloque pertenece esta celda
            bloque_i = i // sqrt_n
            bloque_j = j // sqrt_n
            
            # Decidir color basado en el bloque para crear patrón
            color_base = color_calculado if (bloque_i + bloque_j) % 2 == 0 else color_calculado_alt
            
            # Dibujar el rectángulo de fondo de la celda
            rect = plt.Rectangle((j, i), 1, 1, fill=True, facecolor=color_base,
                               edgecolor=color_borde_celda, linewidth=0.5)
            ax.add_patch(rect)
    
    # Dibujar líneas del grid
    # Líneas finas para todas las celdas
    for i in range(n + 1):
        lw = 3 if i % sqrt_n == 0 else 0.5
        color = color_borde_bloque if i % sqrt_n == 0 else color_borde_celda
        ax.axhline(i, color=color, linewidth=lw)
        ax.axvline(i, color=color, linewidth=lw)
    
    # Verificar si la matriz tiene valores incorrectos (fuera del rango 1-9)
    tiene_valores_incorrectos = False
    for i in range(n):
        for j in range(n):
            if matriz[i, j] < 0 or matriz[i, j] > 9 or (matriz[i, j] != int(matriz[i, j]) and matriz[i, j] != 0):
                tiene_valores_incorrectos = True
                print(f"¡Advertencia! Valor no válido en posición [{i},{j}]: {matriz[i, j]}")
    
    if tiene_valores_incorrectos:
        print("La matriz contiene valores fuera del rango válido (1-9) o no enteros.")
    
    # Colorear celdas según su estado
    for i in range(n):
        for j in range(n):
            # Obtener el valor actual y verificar si es válido
            valor = matriz[i, j]
            if valor != 0 and valor != int(valor):
                valor = int(valor)  # Intentar convertir a entero si es posible
            
            # Verificar si la celda es fija
            es_fija = matriz_posiciones is not None and matriz_posiciones[i, j] == 1
            
            # Verificar si el valor es válido (no tiene conflictos)
            # Comprobamos si hay duplicados en la fila, columna o subcuadro
            es_valido = True
            
            # Comprobar fila
            fila = matriz[i, :]
            if valor != 0 and np.sum(fila == valor) > 1:
                es_valido = False
                
            # Comprobar columna
            columna = matriz[:, j]
            if valor != 0 and np.sum(columna == valor) > 1:
                es_valido = False
                
            # Comprobar subbloque
            bloque_i = (i // sqrt_n) * sqrt_n
            bloque_j = (j // sqrt_n) * sqrt_n
            subbloque = matriz[bloque_i:bloque_i+sqrt_n, bloque_j:bloque_j+sqrt_n]
            if valor != 0 and np.sum(subbloque == valor) > 1:
                es_valido = False
            
            # Verificar si es un valor corregido
            es_corregido = False
            if not es_fija and matriz_inicial is not None and matriz_inicial[i, j] != 0 and matriz_inicial[i, j] != valor:
                es_corregido = True
            
            # Seleccionar color de fondo basado en el estado
            if es_fija:
                # Celdas fijas tienen un color de fondo distinto
                color_fondo = color_fijo
                color_borde = color_borde_fijo
                lw = 2
            else:
                # Color base que ya establecimos en el patrón de ajedrez
                bloque_i = i // sqrt_n
                bloque_j = j // sqrt_n
                color_fondo = color_calculado if (bloque_i + bloque_j) % 2 == 0 else color_calculado_alt
                color_borde = color_borde_celda
                lw = 0.5
                
            # Si es una celda con conflicto, usar color de error
            if not es_valido and valor != 0:
                # Usar un patrón rayado para indicar conflicto
                rect = plt.Rectangle((j, i), 1, 1, fill=True, facecolor=color_incorrecto, alpha=0.2,
                                   edgecolor=color_incorrecto, linewidth=1, hatch='///')
                ax.add_patch(rect)
            else:
                # Dibujar el rectángulo de fondo de la celda
                rect = plt.Rectangle((j, i), 1, 1, fill=True, facecolor=color_fondo,
                                   edgecolor=color_borde, linewidth=lw)
                ax.add_patch(rect)
            
            # Agregar números con colores según su estado
            if valor != 0:
                # Seleccionar color y estilo del texto
                if es_fija:
                    color_texto = color_texto_fijo
                    weight = 'bold'
                elif es_corregido:
                    color_texto = color_texto_corregido
                    weight = 'bold'
                elif not es_valido:
                    color_texto = color_incorrecto
                    weight = 'normal'
                else:
                    color_texto = color_texto_calculado
                    weight = 'normal'
                
                # Dibujar el número
                ax.text(j + 0.5, i + 0.5, str(int(valor)), 
                       va='center', ha='center', fontsize=16, 
                       fontweight=weight, color=color_texto)
    
    # Añadir leyenda explicativa
    leyenda_items = []
    
    # Añadir rectángulos de ejemplo para cada tipo de celda
    leyenda_items.append(plt.Rectangle((0, 0), 1, 1, fill=True, facecolor=color_fijo, 
                                     edgecolor=color_borde_fijo, linewidth=2, label='Celda fija'))
    
    leyenda_items.append(plt.Rectangle((0, 0), 1, 1, fill=True, facecolor=color_calculado, 
                                     edgecolor=color_borde_celda, linewidth=0.5, label='Celda calculada'))
    
    if matriz_inicial is not None:
        leyenda_items.append(plt.Rectangle((0, 0), 1, 1, fill=True, facecolor=color_calculado, 
                                         edgecolor=color_borde_celda, linewidth=0.5, label='Celda corregida'))
        
        # CORRECCIÓN: Eliminar línea que dibuja un "5" en la posición [1,1] como ejemplo
        # Ahora vamos a crear un ejemplo separado para la leyenda en lugar de dibujarlo en el tablero
        ejemplo_color = color_texto_corregido
        ejemplo_rect = plt.Rectangle((0, 0), 1, 1, fill=True, facecolor='white', 
                                  edgecolor='gray', linewidth=0.5)
        # Agregar un marcador de texto como parte de la leyenda, no en el tablero
        ejemplo_ax = fig.add_axes([0.05, 0.01, 0.05, 0.05], frameon=False)
        ejemplo_ax.text(0.5, 0.5, "5", va='center', ha='center', fontsize=12, 
                      fontweight='bold', color=ejemplo_color)
        ejemplo_ax.set_xticks([])
        ejemplo_ax.set_yticks([])
        ejemplo_ax.set_xlim(0, 1)
        ejemplo_ax.set_ylim(0, 1)
        # Ocultar este eje de la visualización principal, solo lo usamos para la leyenda
        ejemplo_ax.patch.set_alpha(0)
    
    leyenda_items.append(plt.Rectangle((0, 0), 1, 1, fill=True, facecolor=color_incorrecto, alpha=0.2,
                                     edgecolor=color_incorrecto, linewidth=1, hatch='///', label='Conflicto'))
    
    # Colocar leyenda fuera del Sudoku
    legend = ax.legend(handles=leyenda_items, loc='upper center', bbox_to_anchor=(0.5, -0.05),
                      fancybox=True, shadow=True, ncol=4, fontsize=12)
    
    # Añadir un borde a toda la figura
    borde_figura = plt.Rectangle((0, 0), n, n, fill=False, edgecolor=color_borde_bloque, linewidth=4)
    ax.add_patch(borde_figura)
    
    # Verificar si la solución es válida
    es_solucion_valida = True
    errores = 0
    
    for i in range(n):
        # Verificar filas
        if len(set(matriz[i, :])) != n or 0 in matriz[i, :]:
            es_solucion_valida = False
            errores += 1
        
        # Verificar columnas
        if len(set(matriz[:, i])) != n or 0 in matriz[:, i]:
            es_solucion_valida = False
            errores += 1
    
    # Verificar sub-bloques
    for i in range(0, n, sqrt_n):
        for j in range(0, n, sqrt_n):
            subbloque = matriz[i:i+sqrt_n, j:j+sqrt_n].flatten()
            if len(set(subbloque)) != n or 0 in subbloque:
                es_solucion_valida = False
                errores += 1
    
    # Añadir indicador de validez
    if "solución" in titulo.lower() or "resuelto" in titulo.lower():
        validez_color = "#00A878" if es_solucion_valida else "#FF5252"
        estado = "VÁLIDA" if es_solucion_valida else f"INVÁLIDA ({errores} errores)"
        plt.figtext(0.5, -0.03, f"Solución {estado}", 
                   ha="center", fontsize=14, fontweight="bold", color=validez_color,
                   bbox={"facecolor":"white", "edgecolor":validez_color, "alpha":0.9, "pad":5})
    
    # Ajustar el layout
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Guardar imagen si se especifica ruta
    if guardar_como:
        plt.savefig(guardar_como, bbox_inches='tight', dpi=150)
        print(f"Visualización guardada en: {guardar_como}")
    
    # Mostramos el gráfico
    plt.show()


def guardar_resultados(mejor_individuo: Individuo, estadisticas: Dict[str, List[float]], 
                      configuracion: Dict, matriz_inicial: NDArray,
                      nombre_carpeta: str = None):
    """
    Guarda los resultados de la ejecución (configuración, gráfica de estadísticas y solución en imagen).
    
    Args:
        mejor_individuo: El mejor individuo encontrado
        estadisticas: Diccionario con las estadísticas de evolución
        configuracion: Diccionario con la configuración utilizada
        matriz_inicial: Matriz inicial del sudoku
        nombre_carpeta: Nombre de la subcarpeta donde guardar los resultados
    """
    # Crear carpeta 'Resultados' si no existe
    directorio_base = 'Resultados'
    if not os.path.exists(directorio_base):
        os.makedirs(directorio_base)
        print(f"Creada carpeta '{directorio_base}' para guardar resultados.")
    
    # Crear carpeta con el nombre proporcionado por el usuario
    if not nombre_carpeta:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_carpeta = f"sudoku_{timestamp}"
    
    # Ruta completa de la subcarpeta
    directorio_resultados = os.path.join(directorio_base, nombre_carpeta)
    
    # Crear la subcarpeta si no existe
    if not os.path.exists(directorio_resultados):
        os.makedirs(directorio_resultados)
        print(f"Creada subcarpeta '{nombre_carpeta}' para guardar estos resultados.")
    
    # 1. Guardar configuración en formato TXT
    ruta_configuracion = os.path.join(directorio_resultados, "configuracion.txt")
    with open(ruta_configuracion, 'w') as f:
        f.write("=== CONFIGURACIÓN DEL ALGORITMO GENÉTICO ===\n\n")
        
        # Escribir fecha y hora
        f.write(f"Fecha y hora: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Escribir parámetros de población
        f.write("--- Parámetros de población ---\n")
        f.write(f"Tamaño de la población: {configuracion['tam_poblacion']}\n")
        f.write(f"Tamaño de la población élite: {configuracion['tam_elite']}\n")
        f.write(f"Número máximo de generaciones: {configuracion['max_gen']}\n")
        f.write(f"Umbral de estancamiento: {configuracion['umbral_estancamiento']}\n\n")
        
        # Escribir parámetros de mutación
        f.write("--- Parámetros de mutación ---\n")
        f.write(f"Probabilidad de mutación por intercambio: {configuracion['pm1']:.3f}\n")
        f.write(f"Probabilidad inicial de mutación por reinicialización: {configuracion['pm2']:.3f}\n\n")
        
        # Escribir parámetros de cruza
        f.write("--- Parámetros de cruza ---\n")
        f.write(f"Probabilidad de cruza individual: {configuracion['pc1']:.3f}\n")
        f.write(f"Probabilidad de cruza de fila: {configuracion['pc2']:.3f}\n\n")
        
        # Escribir parámetros de selección
        f.write("--- Parámetros de selección ---\n")
        f.write(f"Tamaño del torneo: {configuracion['tam_torneo']}\n")
        f.write(f"Selección estocástica: {'Sí' if configuracion['seleccion_estocastica'] else 'No'}\n\n")
        
        # Escribir parámetros de búsqueda local
        f.write("--- Parámetros de búsqueda local ---\n")
        f.write(f"Búsqueda local habilitada: {'Sí' if configuracion['usar_busqueda_local'] else 'No'}\n")
        f.write(f"Adaptación de estrategias: {'Sí' if configuracion.get('adaptar_estrategias', False) else 'No'}\n\n")
        
        # Escribir resultados
        f.write("--- Resultados ---\n")
        f.write(f"Aptitud final: {mejor_individuo.obtener_aptitud()}\n")
        f.write(f"Solución óptima encontrada: {'Sí' if mejor_individuo.obtener_aptitud() == 0 else 'No'}\n")
        f.write(f"Generaciones calculadas: {len(estadisticas['mejor_aptitud'])}\n")
        
        # Añadir tiempo de ejecución si está disponible
        if 'tiempo_ejecucion' in configuracion:
            minutos = int(configuracion['tiempo_ejecucion'] // 60)
            segundos = int(configuracion['tiempo_ejecucion'] % 60)
            f.write(f"Tiempo de ejecución: {minutos} minutos y {segundos} segundos\n")
    
    # 2. Guardar gráfica de estadísticas
    ruta_estadisticas = os.path.join(directorio_resultados, "estadisticas.png")
    graficar_estadisticas(estadisticas, guardar_como=ruta_estadisticas, mostrar=False)
    
    # 3. Guardar visualización del sudoku
    matriz_posiciones = np.where(matriz_inicial > 0, 1, 0)
    ruta_solucion = os.path.join(directorio_resultados, "solucion.png")
    visualizar_sudoku(mejor_individuo.obtener_matriz_numeros(), matriz_posiciones, matriz_inicial, 
                    titulo="Solución Final", guardar_como=ruta_solucion)
    
    print(f"Resultados guardados en carpeta '{directorio_resultados}':")
    print(f"  - configuracion.txt")
    print(f"  - estadisticas.png")
    print(f"  - solucion.png")

if __name__ == '__main__':
    # Definir la matriz inicial
    SUDOKU_ESTANDAR = np.array([[5, 3, 0, 0, 7, 0, 0, 0, 0],
                                [6, 0, 0, 1, 9, 5, 0, 0, 0],
                                [0, 9, 8, 0, 0, 0, 0, 6, 0],
                                [8, 0, 0, 0, 6, 0, 0, 0, 3],
                                [4, 0, 0, 8, 0, 3, 0, 0, 1],
                                [7, 0, 0, 0, 2, 0, 0, 0, 6],
                                [0, 6, 0, 0 ,0 ,0 ,2 ,8 ,0],
                                [0 ,0 ,0 ,4 ,1 ,9 ,0 ,0 ,5],
                                [0 ,0 ,0 ,0 ,8 ,0 ,0 ,7 ,9]])
    

    SUDOKU_FACIL = np.array([
        [0, 7, 8, 5, 1, 4, 9, 0, 2],
        [5, 6, 0, 0, 0, 9, 3, 0, 0],
        [0, 0, 0, 0, 6, 0, 1, 8, 0],
        [0, 4, 5, 0, 0, 0, 2, 0, 0],
        [8, 3, 2, 7, 5, 1, 4, 0, 0],
        [0, 0, 0, 4, 0, 2, 0, 0, 3],
        [9, 0, 0, 2, 0, 0, 5, 0, 0],
        [0, 0, 0, 1, 3, 8, 0, 4, 0],
        [6, 0, 3, 0, 0, 5, 8, 0, 0]
    ])

    # Intermedio
    SUDOKU_INTERMEDIO = np.array([
        [1, 0, 6, 4, 8, 2, 0, 0, 0],
        [7, 0, 5, 0, 9, 0, 0, 8, 0],
        [8, 3, 2, 7, 5, 0, 0, 6, 0],
        [3, 0, 8, 5, 1, 4, 6, 2, 0],
        [0, 0, 9, 3, 6, 7, 8, 5, 0],
        [5, 0, 1, 0, 0, 0, 7, 0, 0],
        [0, 0, 0, 0, 0, 6, 3, 0, 0],
        [0, 0, 0, 0, 3, 0, 0, 9, 0],
        [0, 1, 0, 9, 0, 0, 0, 0, 8]
    ])

    # Sudoku Experto
    SUDOKU_EXPERTO = np.array([
        [0, 0, 0, 3, 5, 7, 8, 9, 1],
        [3, 5, 0, 0, 0, 0, 7, 0, 0],
        [0, 0, 0, 0, 0, 0, 5, 0, 0],
        [0, 0, 5, 0, 4, 0, 0, 0, 0],
        [0, 7, 0, 9, 8, 0, 0, 5, 0],
        [0, 0, 3, 5, 0, 6, 2, 0, 8],
        [0, 0, 8, 0, 0, 0, 0, 7, 2],
        [0, 0, 0, 4, 2, 0, 1, 8, 0],
        [0, 9, 2, 0, 1, 8, 0, 0, 0]
    ])

    # Sudoku Extremo
    SUDOKU_EXTREMO = np.array([
        [0, 8, 0, 0, 0, 7, 0, 0, 1],
        [0, 0, 6, 0, 0, 3, 0, 0, 0],
        [0, 9, 0, 6, 1, 0, 0, 2, 0],
        [0, 0, 0, 0, 0, 6, 0, 0, 0],
        [0, 0, 9, 4, 8, 0, 0, 0, 3],
        [0, 2, 0, 0, 0, 0, 5, 0, 0],
        [0, 0, 4, 1, 9, 0, 0, 0, 8],
        [0, 0, 0, 0, 7, 0, 0, 0, 0],
        [8, 0, 0, 0, 0, 0, 0, 3, 0]
    ])

    # Sudoku Maestro
    SUDOKU_MAESTRO = np.array([
        [0, 3, 0, 6, 0, 2, 9, 0, 0],
        [0, 0, 0, 5, 0, 9, 0, 0, 0],
        [0, 4, 9, 0, 3, 0, 2, 0, 0],
        [0, 9, 0, 0, 0, 7, 0, 3, 0],
        [2, 1, 8, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 9, 6],
        [0, 0, 2, 0, 1, 5, 0, 0, 8],
        [1, 0, 0, 0, 0, 0, 4, 2, 0],
        [0, 7, 0, 0, 2, 0, 0, 6, 0]
    ])

    # Sudoku Dificil
    SUDOKU_DIFICIL = np.array([
        [0, 0, 6, 3, 0, 0, 0, 0, 0],
        [7, 4, 0, 8, 0, 1, 0, 5, 6],
        [0, 0, 0, 0, 2, 6, 0, 4, 0],
        [0, 6, 0, 0, 0, 0, 0, 0, 0],
        [3, 0, 0, 1, 0, 0, 5, 0, 0],
        [1, 0, 0, 0, 0, 8, 7, 0, 0],
        [4, 0, 2, 0, 8, 7, 0, 1, 0],
        [0, 1, 0, 0, 5, 0, 0, 0, 3],
        [0, 0, 0, 0, 0, 0, 0, 0, 0]  
    ])
    
    matriz = SUDOKU_DIFICIL  # Cambia aquí para probar diferentes sudokus
    print("Sudoku inicial:")
    mostrar_sudoku(matriz)
    
    # Guardar la matriz inicial para comparaciones posteriores
    matriz_inicial = matriz.copy()
    
    # Visualizar el sudoku inicial en forma gráfica
    matriz_posiciones = np.where(matriz > 0, 1, 0)
    visualizar_sudoku(matriz, matriz_posiciones, titulo="Sudoku Inicial")
    
    # Valores por defecto actualizados según lo solicitado
    configuracion = {
        'tam_poblacion': 150,     # Modificado: era 100
        'tam_elite': 50,
        'max_gen': 10000,         # Modificado: era 1000
        'umbral_estancamiento': 100, # Modificado: era 50
        'pm1': 0.3,               # Probabilidad de mutación por intercambio
        'pm2': 0.05,              # Probabilidad inicial de mutación por reinicialización
        'pc1': 0.9,               # Probabilidad de cruce individual
        'pc2': 0.2,               # Probabilidad de cruce de fila
        'tam_torneo': 2,
        'seleccion_estocastica': False,
        'usar_busqueda_local': True
    }
    
    print("\n=== CONFIGURACIÓN DEL ALGORITMO GENÉTICO ===\n")
    
    # Preguntar si se quiere usar configuración rápida
    config_rapida = input("¿Desea usar configuración rápida con valores por defecto? (s/n): ").lower() == 's'
    
    if not config_rapida:
        # Preguntar si se quiere configurar los parámetros de población
        config_poblacion = input("\n¿Desea configurar parámetros de población? (s/n): ").lower() == 's'
        if config_poblacion:
            print("--- Parámetros de población ---")
            configuracion['tam_poblacion'] = int(input(f"Tamaño de la población [{configuracion['tam_poblacion']}]: ") or configuracion['tam_poblacion'])
            configuracion['tam_elite'] = int(input(f"Tamaño de la población élite [{configuracion['tam_elite']}]: ") or configuracion['tam_elite'])
            configuracion['max_gen'] = int(input(f"Número máximo de generaciones [{configuracion['max_gen']}]: ") or configuracion['max_gen'])
            configuracion['umbral_estancamiento'] = int(input(f"Umbral de estancamiento (generaciones sin mejora) [{configuracion['umbral_estancamiento']}]: ") or configuracion['umbral_estancamiento'])
        
        # Preguntar si se quiere configurar los parámetros de mutación
        config_mutacion = input("\n¿Desea configurar parámetros de mutación? (s/n): ").lower() == 's'
        if config_mutacion:
            print("--- Parámetros de mutación ---")
            configuracion['pm1'] = float(input(f"Probabilidad de mutación por intercambio [{configuracion['pm1']}]: ") or configuracion['pm1'])
            configuracion['pm2'] = float(input(f"Probabilidad inicial de mutación por reinicialización [{configuracion['pm2']}]: ") or configuracion['pm2'])
        
        # Preguntar si se quiere configurar los parámetros de cruce
        config_cruce = input("\n¿Desea configurar parámetros de cruce? (s/n): ").lower() == 's'
        if config_cruce:
            print("--- Parámetros de cruce ---")
            configuracion['pc1'] = float(input(f"Probabilidad de cruce individual [{configuracion['pc1']}]: ") or configuracion['pc1'])
            configuracion['pc2'] = float(input(f"Probabilidad de cruce de fila [{configuracion['pc2']}]: ") or configuracion['pc2'])
        
        # Preguntar si se quiere configurar los parámetros de selección
        config_seleccion = input("\n¿Desea configurar parámetros de selección? (s/n): ").lower() == 's'
        if config_seleccion:
            print("--- Parámetros de selección ---")
            configuracion['tam_torneo'] = int(input(f"Tamaño del torneo de selección [{configuracion['tam_torneo']}]: ") or configuracion['tam_torneo'])
            seleccion_estocastica_input = input(f"¿Usar selección estocástica? (s/n) [{'s' if configuracion['seleccion_estocastica'] else 'n'}]: ").lower()
            if seleccion_estocastica_input in ['s', 'n']:
                configuracion['seleccion_estocastica'] = (seleccion_estocastica_input == 's')
        
        # Preguntar si se quiere configurar los parámetros de búsqueda local
        config_busqueda = input("\n¿Desea configurar parámetros de búsqueda local? (s/n): ").lower() == 's'
        if config_busqueda:
            print("--- Parámetros de búsqueda local ---")
            usar_busqueda_local_input = input(f"¿Habilitar búsqueda local? (s/n) [{'s' if configuracion['usar_busqueda_local'] else 'n'}]: ").lower()
            if usar_busqueda_local_input in ['s', 'n']:
                configuracion['usar_busqueda_local'] = (usar_busqueda_local_input == 's')
        

    
    print("\n=== RESUMEN DE CONFIGURACIÓN ===")
    print(f"Población: {configuracion['tam_poblacion']} individuos, {configuracion['tam_elite']} élites, máx {configuracion['max_gen']} generaciones")
    print(f"Mutación: intercambio={configuracion['pm1']:.2f}, reinicialización={configuracion['pm2']:.2f}")
    print(f"Cruce: individual={configuracion['pc1']:.2f}, fila={configuracion['pc2']:.2f}")
    print(f"Selección: torneo de {configuracion['tam_torneo']}, {'estocástica' if configuracion['seleccion_estocastica'] else 'determinística'}")
    print(f"Búsqueda local: {'habilitada' if configuracion['usar_busqueda_local'] else 'deshabilitada'}")
    print(f"Umbral de estancamiento: {configuracion['umbral_estancamiento']} generaciones")
    
    # Confirmar inicio
    iniciar = input("\n¿Iniciar algoritmo genético con esta configuración? (s/n): ").lower() == 's'
    if not iniciar:
        print("Ejecución cancelada por el usuario.")
        exit()
    
    print("\nIniciando algoritmo genético...")
    
    # Iniciar temporizador
    tiempo_inicio = time.time()
    
    # Crear componentes del algoritmo con los parámetros configurados
    mutador = Mutador(pm1=configuracion['pm1'], pm2=configuracion['pm2'])
    busqueda = BusquedaLocal(habilitar_busqueda=configuracion['usar_busqueda_local'])
    cruza = Cruza(pc1=configuracion['pc1'], pc2=configuracion['pc2'])
    
    # Crear la población
    poblacion = Poblacion(
        mutador=mutador,
        busqueda=busqueda,
        cruza=cruza,
        tam_poblacion=configuracion['tam_poblacion'],
        solucion_parcial=matriz,
        tam_elite=configuracion['tam_elite']
    )
    
    # Evolucionar la población con los parámetros configurados
    mejor = poblacion.evolucionar(
        max_generaciones=configuracion['max_gen'],
        umbral_estancamiento=configuracion['umbral_estancamiento']
    )
    
    # Calcular tiempo transcurrido
    tiempo_fin = time.time()
    tiempo_total = tiempo_fin - tiempo_inicio
    
    # Agregar tiempo al diccionario de configuración
    configuracion['tiempo_ejecucion'] = tiempo_total
    
    # Obtener estadísticas
    estadisticas = poblacion.obtener_estadisticas()
    
    # Mostrar tiempo transcurrido
    minutos = int(tiempo_total // 60)
    segundos = int(tiempo_total % 60)
    print(f"\nTiempo de ejecución: {minutos} minutos y {segundos} segundos")
    
    # Graficar estadísticas
    graficar_estadisticas(estadisticas, mostrar=True)
    
    # Mostrar resultado
    if mejor:
        print("\nSolución encontrada:")
        mostrar_sudoku(mejor.obtener_matriz_numeros())
        print(f"Aptitud: {mejor.obtener_aptitud()}")
        
        # Visualizar el sudoku resuelto gráficamente
        titulo_resultado = "Sudoku Resuelto" if mejor.obtener_aptitud() == 0 else "Mejor Solución Encontrada"
        visualizar_sudoku(
            mejor.obtener_matriz_numeros(), 
            matriz_posiciones, 
            matriz_inicial, 
            titulo=titulo_resultado
        )
        
        # Preguntar si se desean guardar los resultados
        guardar = input("¿Desea guardar los resultados? (s/n): ").lower() == 's'
        if guardar:
            nombre = input("Ingrese nombre para la carpeta de resultados: ") or None
            guardar_resultados(mejor, estadisticas, configuracion, matriz_inicial, nombre)
    else:
        print("\nNo se encontró una solución.")
        
    # Verificar si la solución es correcta
    if mejor and mejor.obtener_aptitud() == 0:
        print("\n¡La solución es correcta!")
    elif mejor:
        print("\nLa solución encontrada no es óptima.")