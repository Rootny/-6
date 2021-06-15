# Многоуровневые нейронные сети
импортировать  numpy  как  np
import  matplotlib . pyplot  как  plt
импортировать  нейролаб  как  NL

# создаем точку данных ("y = 2x ^ (2) +8")
min_val  =  - 30  # мин. значения по графику x
max_val  =  30  # макс. значения по графику x
num_points  =  160  # колич. элиментов графика
х  =  нп . linspace ( min_val , max_val , num_points )
у  =  2  *  нп . square ( x ) +  8  # уравнение
у  / =  np . linalg . норма ( у )


данные  =  х . RESHAPE ( num_points , 1 ) # меняет форму массива без изменения самих данных (RESHAPE)
метки  =  y . изменить форму ( num_points , 1 )

plt . рисунок ()
plt . разброс ( данные , метки )
plt . xlabel ( 'Измерение 1' )
plt . ylabel ( 'Измерение 2' )
plt . title ( 'Точки данных' )

neural_net  =  нл . нетто . newff ([[ min_val , max_val ]], [ 10 , 6 , 1 ])
neural_net . trainf  =  nl . поезд . train_gd

error  =  neural_net . поезд ( данные , метки , эпохи  =  1000 , шоу  =  100 , цель  =  0,01 )

вывод  =  neural_net . sim ( данные )
y_pred  =  вывод . изменить форму ( num_points )

plt . рисунок ()
plt . сюжет ( ошибка )
plt . xlabel ( 'Количество эпох' )
plt . ylabel ( 'Ошибка' )
plt . title ( 'Прогресс ошибки обучения' )

x_dense  =  np . linspace ( min_val , max_val , num_points  *  2 )
y_dense_pred  =  нейронная_сеть . сим ( x_dense . Reshape ( x_dense . размер , 1 )). изменить форму ( x_dense . размер )

plt . рисунок ()
plt . сюжет ( x_dense , y_dense_pred , '-' , x , y , '.' , x , y_pred , 'p' )
plt . title ( 'Фактическое против прогнозируемого' )

plt . показать ()
