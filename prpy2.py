# Однослойные нейронные сети
импортировать  numpy  как  np
import  matplotlib . pyplot  как  plt
import  neuroolab  as  nl  # Импорт билиотек

# путь
input_data  =  np . loadtxt ( "/qqq/Pr/neural_simple.txt" )

# Превращаем таблицу в 2 столбца с 2 метками
data  =  input_data [:, 0 : 2 ]
label  =  input_data [:, 2 :]

# График ввода данных
plt . рисунок ()
plt . разброс ( данные [:, 0 ], данные [:, 1 ])
plt . xlabel ( 'Измерение 1' )
plt . ylabel ( 'Измерение 2' )
plt . title ( 'Входные данные' )

# установка min и max измерения
dim1_min , dim1_max  =  данные [:, 0 ]. min (), данные [:, 0 ]. макс ()
dim2_min , dim2_max  =  данные [:, 1 ]. min (), данные [:, 1 ]. макс ()

# Число нейронов
nn_output_layer  =  метки . форма [ 1 ]

# Делаем однослойную сеть
dim1  = [ dim1_min , dim1_max ]
dim2  = [ dim2_min , dim2_max ]
neural_net  =  нл . нетто . newp ([ dim1 , dim2 ], nn_output_layer )

# Задаем эпохи и скорость тренировки
error  =  neural_net . поезд ( данные , метки , эпохи  =  200 , шоу  =  20 , lr  =  0,01 )

# Визуализация графика процесса тренеровки
plt . рисунок ()
plt . сюжет ( ошибка )
plt . xlabel ( 'Количество эпох' )
plt . ylabel ( 'Ошибка обучения' )
plt . title ( 'Прогресс ошибки обучения' )
plt . сетка ()
plt . показать ()

print ( ' \ n Результаты теста:' )
data_test  = [[ 1.5 , 3.2 ], [ 3.6 , 1.7 ], [ 3.6 , 5.7 ], [ 1.6 , 3.9 ]]
для  элемента  в  data_test :
    print ( item , '->' , neural_net . sim ([ item ]) [ 0 ])
