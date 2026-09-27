// Fast Flag marshal panel: 3 zone LEDs + buzzer. Serial 115200.
//   Z,<1-3>,<CLEAR|YELLOW|DOUBLE_YELLOW>
//   G,<GREEN|VSC|SC|RED>
//   M,<text>  (ignored)

const uint8_t ZONE_GREEN[3]  = {6, 10, A0};
const uint8_t ZONE_YELLOW[3] = {7, 13, A1};
const uint8_t PIN_RED  = A2;
const uint8_t PIN_SC   = A3;
const uint8_t PIN_BUZZ = 8;

// 0 = clear, 1 = yellow, 2 = double yellow
uint8_t zones[3] = {0, 0, 0};
// 0 = green, 1 = VSC, 2 = SC, 3 = RED
uint8_t globalFlag = 0;

unsigned long beepUntil = 0;
bool beeping = false;

char buf[40];
uint8_t bufLen = 0;

void beep() {
  digitalWrite(PIN_BUZZ, HIGH);
  beepUntil = millis() + 150;
  beeping = true;
}

void handleLine(char *line) {
  if (line[0] == 'Z' && line[1] == ',') {
    int zone = line[2] - '0';
    if (zone < 1 || zone > 3 || line[3] != ',') return;
    char *f = line + 4;
    uint8_t nf;
    if (strcmp(f, "CLEAR") == 0) nf = 0;
    else if (strcmp(f, "YELLOW") == 0) nf = 1;
    else if (strcmp(f, "DOUBLE_YELLOW") == 0) nf = 2;
    else return;
    if (nf > zones[zone - 1]) beep();
    zones[zone - 1] = nf;
  } else if (line[0] == 'G' && line[1] == ',') {
    char *f = line + 2;
    uint8_t nf;
    if (strcmp(f, "GREEN") == 0) nf = 0;
    else if (strcmp(f, "VSC") == 0) nf = 1;
    else if (strcmp(f, "SC") == 0) nf = 2;
    else if (strcmp(f, "RED") == 0) nf = 3;
    else return;
    if (nf > globalFlag) beep();
    globalFlag = nf;
  }
  // anything else (M lines) is ignored
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
      bufLen = 0;
    }
  }
}

void updateOutputs() {
  unsigned long now = millis();
  bool blinkOn = ((now / 250) % 2) == 0;

  for (uint8_t i = 0; i < 3; i++) {
    digitalWrite(ZONE_GREEN[i], zones[i] == 0 ? HIGH : LOW);
    bool yellow = zones[i] == 1 || (zones[i] == 2 && blinkOn);
    digitalWrite(ZONE_YELLOW[i], yellow ? HIGH : LOW);
  }

  digitalWrite(PIN_RED, globalFlag == 3 ? HIGH : LOW);
  bool scOn = globalFlag == 2 || (globalFlag == 1 && blinkOn);
  digitalWrite(PIN_SC, scOn ? HIGH : LOW);

  if (beeping && (long)(now - beepUntil) >= 0) {
    digitalWrite(PIN_BUZZ, LOW);
    beeping = false;
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

  Serial.begin(115200);
  Serial.println("READY");
}

void loop() {
  readSerial();
  updateOutputs();
}