import krpc
import time
import csv

def main():
    # Параметры целевой орбиты
    target_alt_apoapsis = 408_000       # 408 км
    target_alt_periapsis = 398_000      # 398 км

    conn = krpc.connect(name='Proton-K Launch Autopilot')
    space_center = conn.space_center
    vessel = space_center.active_vessel

    orbit = vessel.orbit
    
    # Готовим файл для записи телеметрии (данных о времени, высоте и скорости)
    with open('telemetry.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Time (s)', 'Altitude (m)', 'VerticalSpeed (m/s)', 'HorizontalSpeed (m/s)'])

        flight_info = vessel.flight(vessel.orbit.body.reference_frame)
        
        # Сбрасываем РСУ и САС (чтобы не мешались на данном этапе :) ), ставим тягу на полную
        vessel.control.rcs = False
        vessel.control.sas = False
        vessel.control.throttle = 1.0
        
        # Устанавливаем автопилот
        vessel.auto_pilot.engage()
        vessel.auto_pilot.target_roll = 0
        vessel.auto_pilot.target_pitch_and_heading(90, 90)

        counter = 5
        while counter > 0:
            print(f"Запуск через {counter}...")
            counter -= 1
            time.sleep(1)

        print("Пуск!")
        vessel.control.activate_next_stage()
        time.sleep(1)
        
        gravity_turn_started = False
        turn_angle_start = 100.0    # скорость, при которой начинаем «наклон» (м/с)
        target_pitch_end = 0        # неообходимый тангаж в конце гравитационного манёвра
        start_time = time.time()
        first_stage_completed = False

        while True:
            t = time.time() - start_time

            altitude = flight_info.mean_altitude
            vertical_speed = flight_info.vertical_speed
            horizontal_speed = flight_info.horizontal_speed
            pitch = vessel.flight().pitch
            heading = vessel.flight().heading

            writer.writerow([f'{t:.3f}',
                             f'{altitude:.3f}',
                             f'{vertical_speed:.3f}',
                             f'{horizontal_speed:.3f}'])

            print(f"Время со старта - {t:.2f} секунд. Высота = {altitude:.2f} м. Верт. скорость = {vertical_speed:.2f} м/c. Гор. скорость = {horizontal_speed:.2f} м/c. Тангаж={pitch:.2f}°. Направление={heading:.2f}°")

            if not first_stage_completed:
                stage_resources = vessel.resources_in_decouple_stage(stage=vessel.control.current_stage - 1, cumulative=False)
                lf = stage_resources.amount('LiquidFuel')

                # Отсоединяем первую ступень, если в ней нет топлива
                if lf < 0.1:
                    first_stage_completed = True
                    print(f"Отстыковка первой ступени! Время со старта - {time.time() - start_time:.2f} секунд.")
                    last_throttle = vessel.control.throttle
                    vessel.control.throttle = 0.0

                    # Делаем небольшую пауза до и после отсоединения ступени, чтобы избежать "неприятные ситуации"
                    time.sleep(0.1)
                    vessel.control.activate_next_stage()
                    time.sleep(0.1)

                    vessel.control.throttle = last_throttle

            # Гравитационный маневр
            if not gravity_turn_started and flight_info.speed > turn_angle_start:
                gravity_turn_started = True
                print("Начинаем гравитационный манёвр...")
            
            if gravity_turn_started:
                speed_range = 1000.0 - turn_angle_start
                current_speed_over = flight_info.speed - turn_angle_start
                frac = min(max(current_speed_over / speed_range, 0), 1)  # от 0 до 1
                new_pitch = 90 - frac * (90 - target_pitch_end)
                
                vessel.auto_pilot.target_pitch_and_heading(new_pitch, 90)

            # Отслеживаем высоту апоцентра
            apoapsis_altitude = orbit.apoapsis_altitude
            if apoapsis_altitude > target_alt_apoapsis * 0.95:
                print("Достигли 95% высоты необходимого апоцентра, сбрасываем тягу и отсоединяем вторую ступень.")
                vessel.control.throttle = 0.0

                time.sleep(0.1)
                print(f"Отстыковка второй ступени! Запись данных телеметрии окончена! Время со старта - {time.time() - start_time:.2f} секунд.")
                vessel.control.activate_next_stage()
                time.sleep(0.1)

                break
            
            time.sleep(0.1)

        # Включаем РСУ, чтобы поддерживать тангаж и наклон
        vessel.control.rcs = True
    
        # Ждем приближения к апоцентру
        print("Ожидаем подъёма до апоцентра для начала округления орбиты...")
        while vessel.orbit.time_to_apoapsis > 50:
            vessel.auto_pilot.target_pitch_and_heading(target_pitch_end, 90)
            print(f"Время, оставшееся до достижения апоцентра: {time.strftime("%Mм %Sс", time.gmtime(vessel.orbit.time_to_apoapsis))}")
            time.sleep(1)

        # Готовимся к округлению орбиты (на всякий случай выравниваем нашу ракету)
        vessel.auto_pilot.target_pitch_and_heading(target_pitch_end, 90)
        print("Начинаем округление орбиты...")

        # Округляем орбиту
        while orbit.periapsis_altitude < target_alt_periapsis:
            vessel.control.throttle = 1.0
            print(f"Текущая высота перицентра: {orbit.periapsis_altitude}, необходимая высота: {target_alt_periapsis}")

            vessel.auto_pilot.target_pitch_and_heading(target_pitch_end, 90)
            time.sleep(0.2)

        vessel.control.throttle = 0.0
        print(f"Орбита достигнута! Приступаю к отделению последней ступени для вывода спутника на орбиту")

        time.sleep(0.1)
        vessel.control.activate_next_stage()
        time.sleep(0.1)

        print("Запускаю системы спутника: солнечные батареи, радиаторы и т. д.")
        vessel.control.toggle_action_group(0)
        vessel.control.toggle_action_group(1)
        vessel.control.toggle_action_group(2)

        print("Спутник успешно выведен на околоземную орбиту! Все системы в норме! Работа автопилота заверешена!")

if __name__ == '__main__':
    main()