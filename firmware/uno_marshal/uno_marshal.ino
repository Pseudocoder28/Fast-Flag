// Fast Flag marshal panel + LCD, single Arduino Uno.
// Serial 115200, newline terminated:
//   Z,<1-3>,<CLEAR|YELLOW|DOUBLE_YELLOW>
//   G,<GREEN|VSC|SC|RED>
//   M,<text, max 16 chars>
// Prints READY when setup is done.

#include <LiquidCrystal.h>
#include <Servo.h>

// LCD pins: RS, E, D4, D5, D6, D7
LiquidCrystal lcd(12, 11, 5, 4, 3, 2);
Servo flagServo;

const uint8_t ZONE_GREEN[3]  = {6, 10, A0};
const uint8_t ZONE_YELLOW[3] = {7, 13, A1};
const uint8_t PIN_RED   = A2;
const uint8_t PIN_SC    = A3;
const uint8_t PIN_SERVO = 9;
const uint8_t PIN_BUZZ  = 8;

const int SERVO_REST    = 0;
const int SERVO_UP      = 90;
const int SERVO_WAVE_LO = 60;
const int SERVO_WAVE_HI = 120;
const unsigned long WAVE_MS      = 3000;  // wave duration after each escalation
const unsigned long WAVE_STEP_MS = 250;
const unsigned long BEEP_MS      = 150;
const unsigned long BLINK_MS     = 250;   // DOUBLE_YELLOW and VSC blink rate

enum ZoneFlag : uint8_t { Z_CLEAR, Z_YELLOW, Z_DOUBLE };
enum GlobalFlag : uint8_t { G_GREEN, G_VSC, G_SC, G_RED };

ZoneFlag zones[3] = {Z_CLEAR, Z_CLEAR, Z_CLEAR};
GlobalFlag globalFlag = G_GREEN;

unsigned long waveUntil = 0;
unsigned long beepUntil = 0;
bool beeping = false;
int lastServoPos = -1;

char buf[40];
uint8_t bufLen = 0;

void setServo(int pos) {
  if (pos != lastServoPos) {
    flagServo.write(pos);
    lastServoPos = pos;
  }
}

void startWave() { waveUntil = millis() + WAVE_MS; }

void beep() {
  digitalWrite(PIN_BUZZ, HIGH);
  beepUntil = millis() + BEEP_MS;
  beeping = true;
}

bool anyFlagged() {
  if (globalFlag != G_GREEN) return true;
  for (uint8_t i = 0; i < 3; i++) {
    if (zones[i] != Z_CLEAR) return true;
  }
  return false;
}

void lcdLine(uint8_t row, const char *text) {
  lcd.setCursor(0, row);
  uint8_t n = 0;
  while (text[n] && n < 16) { lcd.write(text[n]); n++; }
  while (n < 16) { lcd.write(' '); n++; }
}

void showGlobal() {
  switch (globalFlag) {
    case G_GREEN: lcdLine(0, "TRACK GREEN"); break;
    case G_VSC:   lcdLine(0, "VIRTUAL SC");  break;
    case G_SC:    lcdLine(0, "SAFETY CAR");  break;
    case G_RED:   lcdLine(0, "RED FLAG");    break;
  }
}

void handleZone(int zone, const char *flag) {
  if (zone < 1 || zone > 3) { Serial.println(F("ERR zone")); return; }
  ZoneFlag nf;
  if (strcmp(flag, "CLEAR") == 0)              nf = Z_CLEAR;
  else if (strcmp(flag, "YELLOW") == 0)        nf = Z_YELLOW;
  else if (strcmp(flag, "DOUBLE_YELLOW") == 0) nf = Z_DOUBLE;
  else { Serial.println(F("ERR zone flag")); return; }

  ZoneFlag old = zones[zone - 1];
  zones[zone - 1] = nf;
  if (nf > old) { beep(); startWave(); }
}

void handleGlobal(const char *flag) {
  GlobalFlag nf;
  if (strcmp(flag, "GREEN") == 0)    nf = G_GREEN;
  else if (strcmp(flag, "VSC") == 0) nf = G_VSC;
  else if (strcmp(flag, "SC") == 0)  nf = G_SC;
  else if (strcmp(flag, "RED") == 0) nf = G_RED;
  else { Serial.println(F("ERR global flag")); return; }

  GlobalFlag old = globalFlag;
  globalFlag = nf;
  if (nf > old) { beep(); startWave(); }
  showGlobal();
}

void handleLine(char *line) {
  if (line[0] == '\0') return;

  // M first, so the text itself may contain commas
  if (line[0] == 'M' && line[1] == ',') {
    lcdLine(1, line + 2);
    return;
  }

  char *kind = strtok(line, ",");
  if (kind == NULL) return;

  if (strcmp(kind, "Z") == 0) {
    char *z = strtok(NULL, ",");
    char *f = strtok(NULL, ",");
    if (z == NULL || f == NULL) { Serial.println(F("ERR Z format")); return; }
    handleZone(atoi(z), f);
  } else if (strcmp(kind, "G") == 0) {
    char *f = strtok(NULL, ",");
    if (f == NULL) { Serial.println(F("ERR G format")); return; }
    handleGlobal(f);
  } else {
    Serial.println(F("ERR unknown"));
  }
}

void readSerial() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      buf[bufLen] = '\0';
      handleLine(buf);
      bufLen = 0;
    } else if (bufLen < sizeof(buf) - 1) {
      buf[bufLen++] = c;
    } else {
      bufLen = 0;  // line too long, drop it
      Serial.println(F("ERR overflow"));
    }
  }
}

void updateOutputs() {
  unsigned long now = millis();
  bool blinkOn = ((now / BLINK_MS) % 2) == 0;

  for (uint8_t i = 0; i < 3; i++) {
    bool green  = zones[i] == Z_CLEAR;
    bool yellow = zones[i] == Z_YELLOW || (zones[i] == Z_DOUBLE && blinkOn);
    digitalWrite(ZONE_GREEN[i],  green  ? HIGH : LOW);
    digitalWrite(ZONE_YELLOW[i], yellow ? HIGH : LOW);
  }

  digitalWrite(PIN_RED, globalFlag == G_RED ? HIGH : LOW);
  bool scOn = globalFlag == G_SC || (globalFlag == G_VSC && blinkOn);
  digitalWrite(PIN_SC, scOn ? HIGH : LOW);

  if (beeping && (long)(now - beepUntil) >= 0) {
    digitalWrite(PIN_BUZZ, LOW);
    beeping = false;
  }

  if ((long)(waveUntil - now) > 0) {
    setServo(((now / WAVE_STEP_MS) % 2) ? SERVO_WAVE_HI : SERVO_WAVE_LO);
  } else {
    setServo(anyFlagged() ? SERVO_UP : SERVO_REST);
  }
}

void lampTest() {
  const uint8_t pins[8] = {6, 7, 10, 13, A0, A1, A2, A3};
  for (uint8_t i = 0; i < 8; i++) {
    digitalWrite(pins[i], HIGH);
    delay(120);
    digitalWrite(pins[i], LOW);
  }
}

void setup() {
  for (uint8_t i = 0; i < 3; i++) {
    pinMode(ZONE_GREEN[i], OUTPUT);
    pinMode(ZONE_YELLOW[i], OUTPUT);
  }
  pinMode(PIN_RED, OUTPUT);
  pinMode(PIN_SC, OUTPUT);
  pinMode(PIN_BUZZ, OUTPUT);
  digitalWrite(PIN_BUZZ, LOW);

  flagServo.attach(PIN_SERVO);
  setServo(SERVO_REST);

  lcd.begin(16, 2);
  showGlobal();
  lcdLine(1, "FAST FLAG");

  lampTest();  // each LED lights once in order: checks wiring

  Serial.begin(115200);
  Serial.println(F("READY"));
}

void loop() {
  readSerial();
  updateOutputs();
}