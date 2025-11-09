import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QVBoxLayout,
                             QHBoxLayout, QWidget, QPushButton, QComboBox,
                             QLabel, QFileDialog, QTextEdit, QTableWidget,
                             QTableWidgetItem, QMessageBox)
from PyQt5.QtCore import Qt
import datetime
import os

try:
    import seaborn as sns

    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False


class DataVisualizationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.df = None
        self.current_file_path = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Data Visualization App')
        self.setGeometry(100, 100, 1200, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Создаем вкладки
        self.create_stats_tab()
        self.create_correlation_tab()
        self.create_heatmap_tab()
        self.create_line_plot_tab()
        self.create_log_tab()

        self.log_action("Приложение запущено")

    def create_stats_tab(self):
        """1 вкладка - статистика по данным"""
        stats_tab = QWidget()
        layout = QVBoxLayout(stats_tab)

        # Панель управления
        controls_layout = QHBoxLayout()

        self.load_btn = QPushButton('Загрузить файл данных')
        self.load_btn.clicked.connect(self.load_data_file)

        controls_layout.addWidget(self.load_btn)
        controls_layout.addStretch()

        # Информация о файле
        self.file_info_label = QLabel('Файл не загружен')
        self.file_info_label.setStyleSheet("font-weight: bold; color: blue;")

        layout.addLayout(controls_layout)
        layout.addWidget(self.file_info_label)

        # Таблица статистики
        self.stats_table = QTableWidget()
        layout.addWidget(self.stats_table)

        self.tabs.addTab(stats_tab, "Статистика")

    def create_correlation_tab(self):
        """2 вкладка - графики корреляции"""
        correlation_tab = QWidget()
        layout = QVBoxLayout(correlation_tab)

        # Управление
        controls_layout = QHBoxLayout()

        self.corr_btn = QPushButton('Построить графики корреляции')
        self.corr_btn.clicked.connect(self.plot_correlations)

        controls_layout.addWidget(self.corr_btn)
        controls_layout.addStretch()

        # Область для графиков
        self.corr_figure = Figure(figsize=(10, 8))
        self.corr_canvas = FigureCanvas(self.corr_figure)

        layout.addLayout(controls_layout)
        layout.addWidget(self.corr_canvas)

        self.tabs.addTab(correlation_tab, "Графики корреляции")

    def create_heatmap_tab(self):
        """3 вкладка - тепловая карта"""
        heatmap_tab = QWidget()
        layout = QVBoxLayout(heatmap_tab)

        # Управление
        controls_layout = QHBoxLayout()

        self.heatmap_btn = QPushButton('Построить тепловую карту')
        self.heatmap_btn.clicked.connect(self.plot_heatmap)

        controls_layout.addWidget(self.heatmap_btn)
        controls_layout.addStretch()

        # Область для тепловой карты
        self.heatmap_figure = Figure(figsize=(10, 8))
        self.heatmap_canvas = FigureCanvas(self.heatmap_figure)

        layout.addLayout(controls_layout)
        layout.addWidget(self.heatmap_canvas)

        self.tabs.addTab(heatmap_tab, "Тепловая карта")

    def create_line_plot_tab(self):
        """4 вкладка - линейные графики"""
        line_tab = QWidget()
        layout = QVBoxLayout(line_tab)

        # Управление
        controls_layout = QHBoxLayout()

        self.column_combo = QComboBox()
        self.plot_btn = QPushButton('Построить линейный график')
        self.plot_btn.clicked.connect(self.plot_line_chart)

        controls_layout.addWidget(QLabel('Выберите столбец:'))
        controls_layout.addWidget(self.column_combo)
        controls_layout.addWidget(self.plot_btn)
        controls_layout.addStretch()

        # Область для графиков
        self.line_figure = Figure(figsize=(10, 6))
        self.line_canvas = FigureCanvas(self.line_figure)

        layout.addLayout(controls_layout)
        layout.addWidget(self.line_canvas)

        self.tabs.addTab(line_tab, "Линейные графики")

    def create_log_tab(self):
        """5 вкладка - лог действий"""
        log_tab = QWidget()
        layout = QVBoxLayout(log_tab)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)

        # Кнопка очистки лога
        clear_btn = QPushButton('Очистить лог')
        clear_btn.clicked.connect(self.clear_log)

        layout.addWidget(QLabel('Лог действий:'))
        layout.addWidget(self.log_text)
        layout.addWidget(clear_btn)

        self.tabs.addTab(log_tab, "Ход работы")

    def load_data_file(self):
        """Загрузка файла данных (CSV или Excel)"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            'Выберите файл данных',
            '',
            'Data Files (*.csv *.xlsx *.xls);;CSV Files (*.csv);;Excel Files (*.xlsx *.xls);;All Files (*)'
        )

        if file_path:
            try:
                self.current_file_path = file_path
                file_ext = os.path.splitext(file_path)[1].lower()

                if file_ext == '.csv':
                    # Пробуем разные кодировки для CSV
                    encodings = ['utf-8', 'cp1251', 'latin1', 'iso-8859-1']
                    for encoding in encodings:
                        try:
                            self.df = pd.read_csv(file_path, encoding=encoding)
                            break
                        except UnicodeDecodeError:
                            continue
                    else:
                        self.df = pd.read_csv(file_path, encoding='utf-8', errors='ignore')
                elif file_ext in ['.xlsx', '.xls']:
                    self.df = pd.read_excel(file_path)
                else:
                    QMessageBox.warning(self, "Ошибка", "Неподдерживаемый формат файла")
                    return

                # Обновляем информацию о файле
                file_name = os.path.basename(file_path)
                file_info = f"Загружен файл: {file_name} | Строк: {len(self.df):,} | Столбцов: {len(self.df.columns)}"
                self.file_info_label.setText(file_info)

                # Обновляем комбобокс с числовыми столбцами
                self.update_column_combo()

                # Показываем статистику
                self.show_statistics()

                self.log_action(f"Загружен файл: {file_name}")
                QMessageBox.information(self, "Успех", "Данные успешно загружены!")

            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Ошибка загрузки файла: {str(e)}")
                self.log_action(f"Ошибка загрузки файла: {str(e)}")

    def update_column_combo(self):
        """Обновление списка числовых столбцов"""
        if self.df is not None:
            self.column_combo.clear()
            numeric_columns = self.df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                self.column_combo.addItem(col)

    def show_statistics(self):
        """Отображение статистики по данным"""
        if self.df is None:
            return

        try:
            # Базовая статистика
            stats = self.df.describe(include='all').T
            stats['null_count'] = self.df.isnull().sum()
            stats['null_percent'] = (stats['null_count'] / len(self.df) * 100).round(2)
            stats['dtype'] = self.df.dtypes

            # Настраиваем таблицу
            self.stats_table.setRowCount(len(stats))
            self.stats_table.setColumnCount(len(stats.columns) + 1)

            # Заголовки
            headers = ['Столбец'] + list(stats.columns)
            self.stats_table.setHorizontalHeaderLabels(headers)

            # Заполняем данные
            for i, (col_name, row_data) in enumerate(stats.iterrows()):
                self.stats_table.setItem(i, 0, QTableWidgetItem(str(col_name)))
                for j, value in enumerate(row_data):
                    self.stats_table.setItem(i, j + 1, QTableWidgetItem(str(value)))

            self.stats_table.resizeColumnsToContents()
            self.log_action("Отображена статистика по данным")

        except Exception as e:
            QMessageBox.warning(self, "Ошибка", f"Ошибка расчета статистики: {str(e)}")

    def plot_correlations(self):
        """Построение графиков корреляции с использованием Seaborn"""
        if self.df is None:
            QMessageBox.warning(self, "Предупреждение", "Сначала загрузите данные!")
            return

        numeric_df = self.df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) < 2:
            QMessageBox.warning(self, "Предупреждение", "Недостаточно числовых столбцов для анализа корреляции!")
            return

        try:
            self.corr_figure.clear()

            if SEABORN_AVAILABLE:
                # Используем pairplot из seaborn для визуализации парных корреляций
                g = sns.pairplot(numeric_df, diag_kind='hist', corner=False)
                self.corr_figure = g.fig
                self.corr_canvas.figure = self.corr_figure
                self.log_action("Построены графики корреляции (Seaborn pairplot)")
            else:
                # Альтернатива без seaborn - строим scatter matrix
                from pandas.plotting import scatter_matrix
                ax = scatter_matrix(numeric_df, alpha=0.8, figsize=(10, 8), diagonal='hist',
                                    ax=self.corr_figure.add_subplot(111))
                self.log_action("Построены графики корреляции (Scatter matrix)")

            self.corr_canvas.draw()

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка построения графиков корреляции: {str(e)}")
            self.log_action(f"Ошибка построения графиков корреляции: {str(e)}")

    def plot_heatmap(self):
        """Построение тепловой карты корреляций"""
        if self.df is None:
            QMessageBox.warning(self, "Предупреждение", "Сначала загрузите данные!")
            return

        numeric_df = self.df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) < 2:
            QMessageBox.warning(self, "Предупреждение", "Недостаточно числовых столбцов для тепловой карты!")
            return

        try:
            self.heatmap_figure.clear()
            ax = self.heatmap_figure.add_subplot(111)

            # Вычисляем корреляционную матрицу
            corr_matrix = numeric_df.corr()

            if SEABORN_AVAILABLE:
                # Тепловая карта с seaborn
                sns.heatmap(corr_matrix,
                            annot=True,
                            cmap='coolwarm',
                            center=0,
                            square=True,
                            fmt='.3f',
                            linewidths=0.5,
                            cbar_kws={"shrink": 0.8},
                            ax=ax)

                # Улучшаем читаемость подписей
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

            else:
                # Тепловая карта с matplotlib
                im = ax.imshow(corr_matrix.values, cmap='coolwarm', vmin=-1, vmax=1)

                # Добавляем значения корреляции
                for i in range(len(corr_matrix.columns)):
                    for j in range(len(corr_matrix.columns)):
                        ax.text(j, i, f'{corr_matrix.iloc[i, j]:.3f}',
                                ha='center', va='center', color='black', fontsize=8)

                # Настройка осей
                ax.set_xticks(range(len(corr_matrix.columns)))
                ax.set_yticks(range(len(corr_matrix.columns)))
                ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
                ax.set_yticklabels(corr_matrix.columns)

                # Цветовая шкала
                plt.colorbar(im, ax=ax)

            ax.set_title('Тепловая карта корреляций')
            self.heatmap_canvas.draw()
            self.log_action("Построена тепловая карта корреляций")

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка построения тепловой карты: {str(e)}")
            self.log_action(f"Ошибка построения тепловой карты: {str(e)}")

    def plot_line_chart(self):
        """Построение линейного графика для выбранного столбца"""
        if self.df is None:
            QMessageBox.warning(self, "Предупреждение", "Сначала загрузите данные!")
            return

        selected_column = self.column_combo.currentText()
        if not selected_column:
            QMessageBox.warning(self, "Предупреждение", "Выберите числовой столбец!")
            return

        try:
            self.line_figure.clear()
            ax = self.line_figure.add_subplot(111)

            # Строим линейный график
            data = self.df[selected_column].dropna()
            ax.plot(data.values, linewidth=1, color='blue', alpha=0.7)

            # Настройки графика
            ax.set_title(f'Линейный график: {selected_column}')
            ax.set_xlabel('Индекс записи')
            ax.set_ylabel(selected_column)
            ax.grid(True, alpha=0.3)

            # Добавляем статистику в заголовок
            mean_val = data.mean()
            std_val = data.std()
            ax.set_title(f'{selected_column}\n(среднее: {mean_val:.2f}, std: {std_val:.2f})')

            self.line_canvas.draw()
            self.log_action(f"Построен линейный график для столбца: {selected_column}")

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка построения графика: {str(e)}")
            self.log_action(f"Ошибка построения линейного графика: {str(e)}")

    def log_action(self, action):
        """Добавление записи в лог действий"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {action}"
        self.log_text.append(log_entry)

    def clear_log(self):
        """Очистка лога действий"""
        self.log_text.clear()
        self.log_action("Лог очищен")


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    window = DataVisualizationApp()
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()