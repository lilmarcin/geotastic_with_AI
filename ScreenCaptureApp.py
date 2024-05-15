import pyautogui
import cv2
import numpy as np
import time
from ultralytics import YOLO
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QLineEdit, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer, Qt
import threading



class ScreenCaptureApp(QWidget):
    def __init__(self):
        super().__init__()
        self.rect = None
        self.interval = 1
        self.capture_thread = None
        self.stop_event = threading.Event()
        self.model = YOLO('runs/classify/train/weights/best.pt')

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Screen Capture App")

        # Left window
        self.left_label = QLabel("Select an area of ​​the screen by dragging the mouse")
        self.left_label = QLabel("Press ENTER to confirm area")
        self.left_capture_button = QPushButton("Start capture frame")
        self.left_interval_label = QLabel("Enter the interval (seconds)")
        self.left_interval_entry = QLineEdit('1')
        self.left_stop_button = QPushButton("Stop capture")
        
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.left_label)
        left_layout.addWidget(self.left_capture_button)
        left_layout.addWidget(self.left_interval_label)
        left_layout.addWidget(self.left_interval_entry)
        left_layout.addWidget(self.left_stop_button)

        # Right window
        self.right_result_label = QLabel("Country:")
        self.right_image_label = QLabel()
        self.right_image_label.setPixmap(QPixmap("world_map/Default.png"))


        right_layout = QVBoxLayout()
        right_layout.addWidget(self.right_result_label)
        right_layout.addWidget(self.right_image_label)

        # Main layout
        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

        self.setLayout(main_layout)

        # Connect buttons to methods
        self.left_capture_button.clicked.connect(self.start_capture)
        self.left_stop_button.clicked.connect(self.stop_capture)

    def start_capture(self):
        self.interval = int(self.left_interval_entry.text())
        self.rect = self.get_screen_region()

        if self.rect:
            self.left_capture_button.setEnabled(False)
            self.left_stop_button.setEnabled(True)
            self.capture_thread = threading.Thread(target=self.capture_loop)
            self.capture_thread.start()
        else:
            QMessageBox.warning(self, "Error", "No screen area selected.")

    def stop_capture(self):
        self.stop_event.set()
        self.left_capture_button.setEnabled(True)
        self.left_stop_button.setEnabled(False)

    def get_screen_region(self):
        screenshot = pyautogui.screenshot()
        screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        rect = cv2.selectROI("Select Screen Area", screenshot)
        cv2.destroyAllWindows()
        return rect

    def capture_loop(self):
        while not self.stop_event.is_set():
            screenshot = pyautogui.screenshot(region=self.rect)
            screenshot = np.array(screenshot)
            frame = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)
            height, width, channel = screenshot.shape
            bytesPerLine = 3 * width
            qImg = QImage(screenshot.data, width, height, bytesPerLine, QImage.Format_RGB888)
            self.left_label.setPixmap(QPixmap.fromImage(qImg))
            results = self.model(frame)
            names = results[0].names
            probs = results[0].probs.top1
            print(f"Country name: {names[probs]}")
            self.right_result_label.setStyleSheet("font-size: 25px; font-family: Robotic;")
            self.right_result_label.setText(f"Country name: {names[probs]}")
            self.display_image(probs)

            time.sleep(self.interval)

    def display_image(self, value):
            # Mapping of country names to image paths
            images = {
                0: 'world_map/Afghanistan.png',
                1: 'world_map/Albania.png',
                2: 'world_map/Algeria.png',
                3: 'world_map/American Samoa.png',
                4: 'world_map/Andorra.png',
                5: 'world_map/Angola.png',
                6: 'world_map/Anguilla.png',
                7: 'world_map/Antigua and Barbuda.png',
                8: 'world_map/Argentina.png',
                9: 'world_map/Armenia.png',
                10: 'world_map/Aruba.png',
                11: 'world_map/Australia.png',
                12: 'world_map/Austria.png',
                13: 'world_map/Azerbaijan.png',
                14: 'world_map/Bahamas.png',
                15: 'world_map/Bahrain.png',
                16: 'world_map/Bangladesh.png',
                17: 'world_map/Barbados.png',
                18: 'world_map/Belarus.png',
                19: 'world_map/Belgium.png',
                20: 'world_map/Belize.png',
                21: 'world_map/Benin.png',
                22: 'world_map/Bermuda.png',
                23: 'world_map/Bhutan.png',
                24: 'world_map/Bolivia.png',
                25: 'world_map/Bosnia and Herzegovina.png',
                26: 'world_map/Botswana.png',
                27: 'world_map/Brazil.png',
                28: 'world_map/British Virgin Islands.png',
                29: 'world_map/Brunei.png',
                30: 'world_map/Bulgaria.png',
                31: 'world_map/Burkina Faso.png',
                32: 'world_map/Burundi.png',
                33: 'world_map/CA-te d-Ivoire.png',
                34: 'world_map/Cabo Verde.png',
                35: 'world_map/Cambodia.png',
                36: 'world_map/Cameroon.png',
                37: 'world_map/Canada.png',
                38: 'world_map/Cayman Islands.png',
                39: 'world_map/Central African Republic.png',
                40: 'world_map/Chad.png',
                41: 'world_map/Chile.png',
                42: 'world_map/China.png',
                43: 'world_map/Colombia.png',
                44: 'world_map/Comoros.png',
                45: 'world_map/Congo.png',
                46: 'world_map/Cook Islands.png',
                47: 'world_map/Costa Rica.png',
                48: 'world_map/Croatia.png',
                49: 'world_map/Cuba.png',
                50: 'world_map/Curacao.png',
                51: 'world_map/Cyprus.png',
                52: 'world_map/Czechia.png',
                53: 'world_map/Democratic Republic of the Congo.png',
                54: 'world_map/Denmark.png',
                55: 'world_map/Djibouti.png',
                56: 'world_map/Dominica.png',
                57: 'world_map/Dominican Republic.png',
                58: 'world_map/Ecuador.png',
                59: 'world_map/Egypt.png',
                60: 'world_map/El Salvador.png',
                61: 'world_map/Equatorial Guinea.png',
                62: 'world_map/Eritrea.png',
                63: 'world_map/Estonia.png',
                64: 'world_map/Eswatini.png',
                65: 'world_map/Ethiopia.png',
                66: 'world_map/Falkland Islands.png',
                67: 'world_map/Faroe Islands.png',
                68: 'world_map/Fiji.png',
                69: 'world_map/Finland.png',
                70: 'world_map/France.png',
                71: 'world_map/French Polynesia.png',
                72: 'world_map/Gabon.png',
                73: 'world_map/Gambia.png',
                74: 'world_map/Georgia.png',
                75: 'world_map/Germany.png',
                76: 'world_map/Ghana.png',
                77: 'world_map/Gibraltar.png',
                78: 'world_map/Greece.png',
                79: 'world_map/Greenland.png',
                80: 'world_map/Grenada.png',
                81: 'world_map/Guadeloupe.png',
                82: 'world_map/Guam.png',
                83: 'world_map/Guatemala.png',
                84: 'world_map/Guinea.png',
                85: 'world_map/Guinea-Bissau.png',
                86: 'world_map/Guyana.png',
                87: 'world_map/Haiti.png',
                88: 'world_map/Holy See.png',
                89: 'world_map/Honduras.png',
                90: 'world_map/Hong Kong.png',
                91: 'world_map/Hungary.png',
                92: 'world_map/Iceland.png',
                93: 'world_map/India.png',
                94: 'world_map/Indonesia.png',
                95: 'world_map/Iran.png',
                96: 'world_map/Iraq.png',
                97: 'world_map/Ireland.png',
                98: 'world_map/Isle of Man.png',
                99: 'world_map/Israel.png',
                100: 'world_map/Italy.png',
                101: 'world_map/Jamaica.png',
                102: 'world_map/Japan.png',
                103: 'world_map/Jordan.png',
                104: 'world_map/Kazakhstan.png',
                105: 'world_map/Kenya.png',
                106: 'world_map/Kiribati.png',
                107: 'world_map/Kuwait.png',
                108: 'world_map/Kyrgyzstan.png',
                109: 'world_map/Laos.png',
                110: 'world_map/Latvia.png',
                111: 'world_map/Lebanon.png',
                112: 'world_map/Lesotho.png',
                113: 'world_map/Liberia.png',
                114: 'world_map/Libya.png',
                115: 'world_map/Liechtenstein.png',
                116: 'world_map/Lithuania.png',
                117: 'world_map/Luxembourg.png',
                118: 'world_map/Macau.png',
                119: 'world_map/Madagascar.png',
                120: 'world_map/Malawi.png',
                121: 'world_map/Malaysia.png',
                122: 'world_map/Maldives.png',
                123: 'world_map/Mali.png',
                124: 'world_map/Malta.png',
                125: 'world_map/Marshall Islands.png',
                126: 'world_map/Martinique.png',
                127: 'world_map/Mauritania.png',
                128: 'world_map/Mauritius.png',
                129: 'world_map/Mayotte.png',
                130: 'world_map/Mexico.png',
                131: 'world_map/Micronesia.png',
                132: 'world_map/Moldova.png',
                133: 'world_map/Monaco.png',
                134: 'world_map/Mongolia.png',
                135: 'world_map/Montenegro.png',
                136: 'world_map/Montserrat.png',
                137: 'world_map/Morocco.png',
                138: 'world_map/Mozambique.png',
                139: 'world_map/Myanmar.png',
                140: 'world_map/Namibia.png',
                141: 'world_map/Nauru.png',
                142: 'world_map/Nepal.png',
                143: 'world_map/Netherlands.png',
                144: 'world_map/Netherlands Antilles.png',
                145: 'world_map/New Zealand.png',
                146: 'world_map/Nicaragua.png',
                147: 'world_map/Niger.png',
                148: 'world_map/Nigeria.png',
                149: 'world_map/Niue.png',
                150: 'world_map/North Korea.png',
                151: 'world_map/North Macedonia.png',
                152: 'world_map/Northern Mariana Islands.png',
                153: 'world_map/Norway.png',
                154: 'world_map/Oman.png',
                155: 'world_map/Pakistan.png',
                156: 'world_map/Palau.png',
                157: 'world_map/Palestine State.png',
                158: 'world_map/Panama.png',
                159: 'world_map/Papua New Guinea.png',
                160: 'world_map/Paraguay.png',
                161: 'world_map/Peru.png',
                162: 'world_map/Philippines.png',
                163: 'world_map/Poland.png',
                164: 'world_map/Portugal.png',
                165: 'world_map/Puerto Rico.png',
                166: 'world_map/Qatar.png',
                167: 'world_map/Reunion.png',
                168: 'world_map/Romania.png',
                169: 'world_map/Russia.png',
                170: 'world_map/Rwanda.png',
                171: 'world_map/Saint Barthelemy.png',
                172: 'world_map/Saint Helena.png',
                173: 'world_map/Saint Kitts and Nevis.png',
                174: 'world_map/Saint Lucia.png',
                175: 'world_map/Saint Pierre and Miquelon.png',
                176: 'world_map/Saint Vincent and the Grenadines.png',
                177: 'world_map/Samoa.png',
                178: 'world_map/San Marino.png',
                179: 'world_map/Sao Tome and Principe.png',
                180: 'world_map/Saudi Arabia.png',
                181: 'world_map/Senegal.png',
                182: 'world_map/Serbia.png',
                183: 'world_map/Seychelles.png',
                184: 'world_map/Sierra Leone.png',
                185: 'world_map/Singapore.png',
                186: 'world_map/Sint Maarten.png',
                187: 'world_map/Slovakia.png',
                188: 'world_map/Slovenia.png',
                189: 'world_map/Solomon Islands.png',
                190: 'world_map/Somalia.png',
                191: 'world_map/South Africa.png',
                192: 'world_map/South Georgia and The South Sandwich Islands.png',
                193: 'world_map/South Korea.png',
                194: 'world_map/South Sudan.png',
                195: 'world_map/Spain.png',
                196: 'world_map/Sri Lanka.png',
                197: 'world_map/Sudan.png',
                198: 'world_map/Suriname.png',
                199: 'world_map/Sweden.png',
                200: 'world_map/Switzerland.png',
                201: 'world_map/Syria.png',
                202: 'world_map/Taiwan.png',
                203: 'world_map/Tajikistan.png',
                204: 'world_map/Tanzania.png',
                205: 'world_map/Thailand.png',
                206: 'world_map/Timor-Leste.png',
                207: 'world_map/Togo.png',
                208: 'world_map/Tokelau.png',
                209: 'world_map/Tonga.png',
                210: 'world_map/Trinidad and Tobago.png',
                211: 'world_map/Tunisia.png',
                212: 'world_map/Turkey.png',
                213: 'world_map/Turkmenistan.png',
                214: 'world_map/Turks and Caicos.png',
                215: 'world_map/Tuvalu.png',
                216: 'world_map/Uganda.png',
                217: 'world_map/Ukraine.png',
                218: 'world_map/United Arab Emirates.png',
                219: 'world_map/United Kingdom.png',
                220: 'world_map/United States of America.png',
                221: 'world_map/Uruguay.png',
                222: 'world_map/Uzbekistan.png',
                223: 'world_map/Vanuatu.png',
                224: 'world_map/Venezuela.png',
                225: 'world_map/Vietnam.png',
                226: 'world_map/Wallis and Futuna.png',
                227: 'world_map/Western Sahara.png',
                228: 'world_map/Yemen.png',
                229: 'world_map/Zambia.png',
                230: 'world_map/Zimbabwe.png'
            }


            if value in images:
                image_path = images[value]
                pixmap = QPixmap(image_path)
                self.right_image_label.setPixmap(pixmap)
            else:
                self.right_image_label.setPixmap(QPixmap("world_map/Default.png"))

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.stop_capture()
            self.close()

def main():
    app = QApplication(sys.argv)
    main_window = ScreenCaptureApp()
    main_window.setWindowTitle("Screen Capture App")
    main_window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
