import sys
import subprocess
import platform
import datetime
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker, declarative_base
from time import sleep
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QTextEdit,
                             QSpinBox, QCheckBox, QGroupBox, QGridLayout)
from PyQt6.QtCore import QThread, pyqtSignal

# Настройка базы данных и SQLAlchemy
Base = declarative_base()


# Модель для таблицы результатов
class BenchmarkResult(Base):
    __tablename__ = 'benchmark_results'

    id = sa.Column(sa.Integer, primary_key=True)
    system_name = sa.Column(sa.String)
    test_datetime = sa.Column(sa.DateTime)
    program_type = sa.Column(sa.String)  # MPI или OpenMP
    num_threads = sa.Column(sa.Integer)
    test_word = sa.Column(sa.String)
    brute_force_time = sa.Column(sa.Float)
    total_execution_time = sa.Column(sa.Float)
    additional_param = sa.Column(sa.String)  # версия программы


# Создание базы данных (если не существует)
engine = sa.create_engine('sqlite:///benchmark_results.db')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()
last_db_update = datetime.datetime.now()


class Worker(QThread):
    update_output = pyqtSignal(str)
    update_status = pyqtSignal(str)

    def __init__(self, command, program_type, version, word, num_threads, chunk_size=0):
        super().__init__()
        self.command = command
        self.program_type = program_type
        self.version = version
        self.word = word
        self.num_threads = num_threads
        self.chunk_size = chunk_size

    def run(self):
        try:
            self.update_status.emit(
                f"Running {self.program_type.upper()} with word: {self.word}, version: {self.version}, threads: {self.num_threads}, chunk: {self.chunk_size}")
            result = subprocess.run(self.command, capture_output=True, text=True)
            self.update_output.emit(result.stdout)

            if result.returncode != 0:
                self.update_output.emit(f"Error: Program exited with code {result.returncode}")
            else:
                self.update_output.emit(f"Finished {self.word} on {self.program_type.upper()} version {self.version}")

            # Запись результата в базу данных
            self.save_result(result.stdout)
        except subprocess.CalledProcessError as e:
            self.update_output.emit(f"Error: {e}")

    def save_result(self, output):
        global last_db_update
        system_name = platform.node()
        test_datetime = datetime.datetime.now()

        # Извлечение времени из вывода программы
        brute_force_time = float(
            [line.split(": ")[1].split(" ")[0] for line in output.splitlines() if "Brute-force time" in line][0])
        total_time = float(
            [line.split(": ")[1].split(" ")[0] for line in output.splitlines() if "Total execution time" in line][0])

        if datetime.datetime.now() - last_db_update < datetime.timedelta(milliseconds=5000):
            sleep(0.5)

        last_db_update = datetime.datetime.now()
        result = BenchmarkResult(
            system_name=system_name,
            test_datetime=test_datetime,
            program_type="MPI" if self.program_type == "mpi" else "OpenMP",
            num_threads=self.num_threads,
            test_word=self.word,
            brute_force_time=brute_force_time,
            total_execution_time=total_time,
            additional_param=self.version
        )

        session.add(result)
        session.commit()
        print(f"Test result saved for {self.program_type.upper()}")


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        # Создание основного лейаута с двумя колонками
        main_layout = QHBoxLayout()

        # Левая колонка для ввода данных
        left_layout = QVBoxLayout()

        # Поля для ввода количества процессов (от и до)
        self.processes_label = QLabel("Number of Processes (from, to):")
        self.processes_from_input = QSpinBox()
        self.processes_from_input.setMinimum(1)
        self.processes_to_input = QSpinBox()
        self.processes_to_input.setMinimum(1)
        left_layout.addWidget(self.processes_label)
        left_layout.addWidget(self.processes_from_input)
        left_layout.addWidget(self.processes_to_input)

        # Поле для ввода тестовых слов
        self.words_label = QLabel("Test Words (comma separated):")
        self.words_input = QLineEdit()
        left_layout.addWidget(self.words_label)
        left_layout.addWidget(self.words_input)

        # Версии OpenMP (группа флажков)
        openmp_group_box = QGroupBox("OpenMP Versions")
        openmp_layout = QVBoxLayout()
        self.openmp_version_checkboxes = []
        for version in ["dynamic", "static", "auto", "guided"]:
            checkbox = QCheckBox(version)
            self.openmp_version_checkboxes.append(checkbox)
            openmp_layout.addWidget(checkbox)
        openmp_group_box.setLayout(openmp_layout)
        left_layout.addWidget(openmp_group_box)

        # Версии MPI (группа флажков)
        mpi_group_box = QGroupBox("MPI Versions")
        mpi_layout = QVBoxLayout()
        self.mpi_version_checkboxes = []
        for version in ["10k", "50k", "old",]:
            checkbox = QCheckBox(version)
            self.mpi_version_checkboxes.append(checkbox)
            mpi_layout.addWidget(checkbox)
        mpi_group_box.setLayout(mpi_layout)
        left_layout.addWidget(mpi_group_box)

        # Поле для ввода чанков OpenMP
        self.chunk_label = QLabel("OpenMP Chunk Size:")
        self.chunk_input = QSpinBox()
        left_layout.addWidget(self.chunk_label)
        left_layout.addWidget(self.chunk_input)

        # Кнопка для запуска теста
        self.run_button = QPushButton("Run Benchmark")
        self.run_button.clicked.connect(self.run_benchmark)
        left_layout.addWidget(self.run_button)

        # Добавляем левую колонку в основной лейаут
        main_layout.addLayout(left_layout)

        # Правая колонка для вывода данных
        right_layout = QVBoxLayout()

        # Поле для вывода статуса
        self.status_label = QLabel("Status:")
        self.status_output = QTextEdit()
        self.status_output.setReadOnly(True)
        right_layout.addWidget(self.status_label)
        right_layout.addWidget(self.status_output)

        # Поле для вывода запускаемых процессов
        self.output_label = QLabel("Output:")
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        right_layout.addWidget(self.output_label)
        right_layout.addWidget(self.output_text)

        # Добавляем правую колонку в основной лейаут
        main_layout.addLayout(right_layout)

        self.setLayout(main_layout)
        self.setWindowTitle('Benchmark Runner')
        self.resize(800, 600)

    def run_benchmark(self):
        processes_from = self.processes_from_input.value()
        processes_to = self.processes_to_input.value()
        words = self.words_input.text().split(", ")

        # Собираем выбранные версии OpenMP
        openmp_versions = [checkbox.text() for checkbox in self.openmp_version_checkboxes if checkbox.isChecked()]

        # Собираем выбранные версии MPI
        mpi_versions = [checkbox.text() for checkbox in self.mpi_version_checkboxes if checkbox.isChecked()]

        chunk_size = self.chunk_input.value()

        # Выполнение MPI команд
        for word in words:
            for mpi_version in mpi_versions:
                for num_threads in range(processes_from, processes_to + 1):
                    command = ["mpirun", "-np", str(num_threads), f"./md5_mpi_{mpi_version}", word]
                    worker = Worker(command, "mpi", mpi_version, word, num_threads)
                    worker.update_output.connect(self.display_output)
                    worker.update_status.connect(self.update_status)
                    worker.start()
                    worker.wait()  # Ожидание завершения потока перед запуском следующего

        # Выполнение OpenMP команд
        for word in words:
            for openmp_version in openmp_versions:
                for num_threads in range(processes_from, processes_to + 1):
                    command = [
                        "bash", "-c", f"OMP_NUM_THREADS={num_threads}",
                        f"OMP_SCHEDULE={openmp_version},{chunk_size}" if chunk_size else f"OMP_SCHEDULE={openmp_version}",
                        f"./md5_openmp {word}"
                    ]
                    worker = Worker(command, "openmp", openmp_version, word, num_threads, chunk_size)
                    worker.update_output.connect(self.display_output)
                    worker.update_status.connect(self.update_status)
                    worker.start()
                    worker.wait()  # Ожидание завершения потока перед запуском следующего

    def display_output(self, output):
        self.output_text.append(output)

    def update_status(self, status):
        self.status_output.append(status)

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
