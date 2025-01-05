import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ----------------------------------------------------------------------------
#       Константы
# ----------------------------------------------------------------------------

G = 6.67430e-11                       # Гравитационная постоянная
C = 0.5                               # Коэффициент сопротивления
T = 300                               # Абсолютная температура (в Кельвинах)

M_earth = 5.9742e24                   # Масса земли (в кг)
R_earth = 6378000                     # Радиус земли (в метрах)
M_molar_mass = 0.029                  # Молярная масса воздуха (в кг/моль)
R_gas_const = 8.31                    # Универсальная газовая постоянная

p_0 = 101330                          # Атмосферное давление на уровне моря (в Па)
e = np.e                              # Число Эйлера (2,7182...)
g = 9.81                              # Ускорение свободного падения (в м/c^2)

time_step = 0.1                       # Шаг вычислений во времени

# ----------------------------------------------------------------------------
#       Константы, относясящиеся к ракете (берем из KSP)
# ----------------------------------------------------------------------------

m_0 = 420347                          # Масса ракеты на момент запуска (в кг)
m_1 = 191902                          # Масса ракеты без первой ступени (в кг)

             #     ЖТ   + Окислитель
fuel_consump_1 = 44.407 + 54.275      # Расход топлива для одного двигателя первой ступени (в кг/c)
fuel_consump_2 = 18.642 + 22.784      # Расход топлива для одного двигателя второй ступени (в кг/c)

mu_1 = fuel_consump_1 * 6             # Расход топлива для 6 двигателей РД-275 первой ступени
mu_2 = fuel_consump_2 * 4             # Расход топлива для 4 двигателей РД-0212 второй ступени

F_thrust_1 = 1500 * 6 * 1e3           # Общая тяга первой ступени (в Ньютонах)
F_thrust_2 = 650  * 4 * 1e3           # Общая тяга второй ступени (в Ньютонах)

alpha_0 = np.pi / 2                   # начальный угол тангажа, ракета направлена вертикально вверх (в радианах)
alpha_1 = np.deg2rad(46.83)           # угол тангажа после отделения первой ступени, взят из симуляции KSP (в радианах)
delta_alpha = -np.deg2rad(0.45)       # скорость изменения угла тангада (в рад/с)
alpha_min = np.deg2rad(0)             # минимальный угол тангажа по KSP, равен 10° (в радианах)
alpha_max = np.deg2rad(350)           # минимальный угол тангажа по KSP, равен 350° = -10° (в радианах)

first_stage_time = 55.19              # Время работы первой ступени (в секундах)
second_stage_time = 169.34 - 55.19    # Время работы второй ступени (в секундах)

d_rocket = 7.1                        # Диаметр ракеты (в метрах)
S = np.pi * ((d_rocket / 2) ** 2)     # Площадь поперечного сечения ракеты (в м^2)


# ----------------------------------------------------------------------------
#       Всмопогательные функции для физической модели
# ----------------------------------------------------------------------------

def calc_current_mass(default_mass, fuel_consumption, time_passed_from_burn_start):
    return (default_mass - fuel_consumption * time_passed_from_burn_start)


def calc_gravitation_force(rocket_mass, altitude):
    return ((G * M_earth * rocket_mass) / ((R_earth + altitude) ** 2))


def calc_air_resistance_force(altitude, speed):
    def calc_atmospheric_pressure():
        return (p_0 * np.exp((-M_molar_mass * g * altitude) / (R_gas_const * T)))


    def calc_air_density():
        return ((calc_atmospheric_pressure() * M_molar_mass) / (R_gas_const * T))


    return ((C * S * (speed ** 2) * calc_air_density()) / 2)


# ----------------------------------------------------------------------------
#       Функции для правых частей системы ОДУ (первая и вторая ступени)
# ----------------------------------------------------------------------------

def derivatives_stage_1(time, altitude, speed_Ox, speed_Oy):
    # Масса и "сырой" угол
    mass = calc_current_mass(m_0, mu_1, time)
    alpha = np.clip(alpha_0 + delta_alpha * time, alpha_min, alpha_max)

    # Скорость и высота
    speed = np.sqrt(speed_Ox**2 + speed_Oy**2)

    air_resistance = calc_air_resistance_force(altitude, speed)

    # Силы сопротивления воздуха (по Ox и Oy)
    F_air_res_x = air_resistance * np.cos(alpha)
    F_air_res_y = air_resistance * np.sin(alpha)

    # Сила гравитации
    F_grav = calc_gravitation_force(mass, altitude)

    # Производные скоростей (скорости по осям Ox и Oy) 
    dvxdt = (F_thrust_1 * np.cos(alpha) - F_air_res_x) / mass
    dvydt = (F_thrust_1 * np.sin(alpha) - F_air_res_y - F_grav) / mass

    return dvxdt, dvydt


def derivatives_stage_2(time, altitude, speed_Ox, speed_Oy):
    # Рассчитываем массу ракеты и угол тангажа на текущее время
    mass = calc_current_mass(m_1, mu_2, time)
    alpha = np.clip(alpha_1 + delta_alpha * time, alpha_min, alpha_max)

    speed = np.sqrt(speed_Ox ** 2 + speed_Oy ** 2)

    # Сила сопротивления воздуха
    air_resistance = calc_air_resistance_force(altitude, speed)

    # Силы сопротивления воздуха (по Ox и Oy)
    F_air_res_x = air_resistance * np.cos(alpha)
    F_air_res_y = air_resistance * np.sin(alpha)

    # Сила гравитации
    F_grav = calc_gravitation_force(mass, altitude)

    # Производные скоростей (скорости по осям Ox и Oy) 
    dvxdt = (F_thrust_2 * np.cos(alpha) - F_air_res_x) / mass
    dvydt = (F_thrust_2 * np.sin(alpha) - F_air_res_y - F_grav) / mass

    return dvxdt, dvydt


# ----------------------------------------------------------------------------
#       Функции для интегрирования
# ----------------------------------------------------------------------------

def euler_solve_stage1(time_stage_start, time_stage_end, x0, y0, vx0, vy0):
    n_steps = int((time_stage_end - time_stage_start) / time_step) + 1

    t_vals  = np.zeros(n_steps)
    x_vals  = np.zeros(n_steps)
    y_vals  = np.zeros(n_steps)
    vx_vals = np.zeros(n_steps)
    vy_vals = np.zeros(n_steps)

    # Начальные условия
    t_vals[0]  = time_stage_start
    x_vals[0]  = x0
    y_vals[0]  = y0
    vx_vals[0] = vx0
    vy_vals[0] = vy0

    for i in range(n_steps - 1):
        t_i  = t_vals[i]
        x_i  = x_vals[i]
        y_i  = y_vals[i]
        vx_i = vx_vals[i]
        vy_i = vy_vals[i]

        dvxdt, dvydt = derivatives_stage_1(t_i, y_i, vx_i, vy_i)

        x_vals[i + 1]  = x_i  + vx_vals[i] * time_step
        y_vals[i + 1]  = y_i  + vy_vals[i] * time_step
        vx_vals[i + 1] = vx_i + dvxdt * time_step
        vy_vals[i + 1] = vy_i + dvydt * time_step
        t_vals[i + 1]  = t_i  + time_step

    return t_vals, x_vals, y_vals, vx_vals, vy_vals


def euler_solve_stage2(time_stage_start, time_stage_end, x0, y0, vx0, vy0):
    n_steps = int((time_stage_end - time_stage_start) / time_step) + 1

    t_vals  = np.zeros(n_steps)
    x_vals  = np.zeros(n_steps)
    y_vals  = np.zeros(n_steps)
    vx_vals = np.zeros(n_steps)
    vy_vals = np.zeros(n_steps)

    t_vals[0]  = time_stage_start
    x_vals[0]  = x0
    y_vals[0]  = y0
    vx_vals[0] = vx0
    vy_vals[0] = vy0

    # Время, с которого начинается вторая ступень
    stage2_t0 = time_stage_start

    for i in range(n_steps - 1):
        t_i  = t_vals[i]
        x_i  = x_vals[i]
        y_i  = y_vals[i]
        vx_i = vx_vals[i]
        vy_i = vy_vals[i]

        # Локальное время (внутри второй ступени)
        t_stage2 = t_i - stage2_t0

        dvxdt, dvydt = derivatives_stage_2(t_stage2, y_i, vx_i, vy_i)

        x_vals[i + 1]  = x_i  + vx_vals[i] * time_step
        y_vals[i + 1]  = y_i  + vy_vals[i] * time_step
        vx_vals[i + 1] = vx_i + dvxdt * time_step
        vy_vals[i + 1] = vy_i + dvydt * time_step
        t_vals[i + 1]  = t_i  + time_step

    return t_vals, x_vals, y_vals, vx_vals, vy_vals


# ----------------------------------------------------------------------------
#       Отображение данных на графиках
# ----------------------------------------------------------------------------

def draw_plot(
        time,                       # Время по матмодели
        model_altitude,             # Высота по матмодели
        model_speed_x,              # Скорость Ox по матмодели
        model_speed_y,              # Скорость Oy по матмодели
        time_simulation,            # Время по KSP 
        simulation_altitude,        # Высота по симуляции в KSP
        simulation_speed_x,         # Скорость Ox по симуляции в KSP
        simulation_speed_y          # Скорость Oy по симуляции в KSP
    ):
    # Скорость по модели
    model_speed  = np.sqrt(model_speed_x**2 + model_speed_y**2)

    # Скорость по симуляции KSP
    simulation_speed = np.sqrt(simulation_speed_x**2 + simulation_speed_y**2)  

    # Так как моменты времени модели и симуляции могут быть разными, мы должны их проинтерполировать
    altitude_model_ksp_times = np.interp(time_simulation, time, model_altitude)
    speed_model_ksp_times  = np.interp(time_simulation, time, model_speed)

    # Расчет погрешности (разницы между симуляцией KSP и моделью)
    altitude_diff =  simulation_altitude - altitude_model_ksp_times
    speed_diff  = simulation_speed - speed_model_ksp_times

    _, axes = plt.subplots(2, 3, figsize=(21, 6))  # Увеличенная ширина для 5 (6) графиков

    # График высоты
    axes[0,0].plot(time, model_altitude, label="Высота (модель)", color="blue")
    axes[0,0].plot(time_simulation, simulation_altitude, label="Высота (KSP)", color="red")
    axes[0,0].set_title("Зависимость высоты от времени")
    axes[0,0].set_xlabel("Время, с")
    axes[0,0].set_ylabel("Высота, м")
    axes[0,0].grid(True)
    axes[0,0].legend()

    # График скорости
    axes[0,1].plot(time, model_speed, label="Скорость (модель)", color="green")
    axes[0,1].plot(time_simulation, simulation_speed, label="Скорость (KSP)", color="red")
    axes[0,1].set_title("Зависимость скорости от времени")
    axes[0,1].set_xlabel("Время, с")
    axes[0,1].set_ylabel("Скорость, м/с")
    axes[0,1].grid(True)
    axes[0,1].legend()    

    # График погрешности
    axes[0,2].plot(time_simulation, altitude_diff, label="ΔВысота = KSP - модель", color="blue")
    axes[0,2].plot(time_simulation, speed_diff,  label="ΔСкорость = KSP - модель", color="red")
    axes[0,2].set_title("Погрешность (разница модели и KSP)")
    axes[0,2].set_xlabel("Время, с")
    axes[0,2].set_ylabel("Разница, м / м/с")
    axes[0,2].grid(True)
    axes[0,2].legend()

    axes[1,0].plot(time, model_speed_x, label="Скорость по Ox (модель)", color="green")
    axes[1,0].plot(time_simulation, simulation_speed_x, label="Скорость по Ox (KSP)", color="red")
    axes[1,0].set_title("Зависимость скорости по Ox от времени")
    axes[1,0].set_xlabel("Время, с")
    axes[1,0].set_ylabel("Скорость по Ox, м/с")
    axes[1,0].grid(True)
    axes[1,0].legend()

    axes[1,1].plot(time, model_speed_y, label="Скорость по Oy (модель)", color="green")
    axes[1,1].plot(time_simulation, simulation_speed_y, label="Скорость по Oy (KSP)", color="red")
    axes[1,1].set_title("Зависимость скорости по Oy от времени")
    axes[1,1].set_xlabel("Время, с")
    axes[1,1].set_ylabel("Скорость по Oy, м/с")
    axes[1,1].grid(True)
    axes[1,1].legend()

    axes[1,2].set_visible(False)

    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------------
#       Решение
# ----------------------------------------------------------------------------

def main():
    # Решаем систему дифф. уравнений для первой ступени
    t_stage1, x_stage1, y_stage1, vx_stage1, vy_stage1 = euler_solve_stage1(
        time_stage_start = 0.0,
        time_stage_end   = first_stage_time,
        x0               = 0.0,
        y0               = 0.0,
        vx0              = 0.0,
        vy0              = 0.0
    )

    # Начальные условия для второй ступени равным последним значениям от первой ступени
    x1_end  = x_stage1[-1]
    y1_end  = y_stage1[-1]
    vx1_end = vx_stage1[-1]
    vy1_end = vy_stage1[-1]
    t1_end  = t_stage1[-1]

    # Решаем систему дифф. уравнений для первой ступени
    t_stage2, x_stage2, y_stage2, vx_stage2, vy_stage2 = euler_solve_stage2(
        time_stage_start = t1_end,
        time_stage_end   = t1_end + second_stage_time,
        x0               = x1_end,
        y0               = y1_end,
        vx0              = vx1_end,
        vy0              = vy1_end
    )

    # Объединяем результаты решения первой и второй системы уравнений
    t_full = np.concatenate((t_stage1, t_stage2[1:]))
    x_full = np.concatenate((x_stage1, x_stage2[1:]))
    y_full = np.concatenate((y_stage1, y_stage2[1:]))
    vx_full = np.concatenate((vx_stage1, vx_stage2[1:]))
    vy_full = np.concatenate((vy_stage1, vy_stage2[1:]))

    # ----------------------------------------------------------------------------
    #       Чтение телеметрии из симуляции KSP
    # ----------------------------------------------------------------------------

    df = pd.read_csv("../Telemetry/telemetry.csv")

    time_ksp            = df.iloc[:, 0].values        # Time
    altitude_ksp        = df.iloc[:, 1].values        # Altitude
    verticalspeed_ksp   = df.iloc[:, 2].values        # VerticalSpeed
    horizontalspeed_ksp = df.iloc[:, 3].values        # HorizontalSpeed  

    # ----------------------------------------------------------------------------
    #       Построение графиков
    # ----------------------------------------------------------------------------

    draw_plot(
        t_full,
        y_full,
        vx_full,
        vy_full,
        time_ksp,
        altitude_ksp,
        horizontalspeed_ksp,
        verticalspeed_ksp
    )
    

if __name__ == "__main__":
    main()
