#include <Arduino.h>
#include <Wire.h>
#include <Modulino.h>
#include "Arduino_RouterBridge.h"

ModulinoPixels leds;

void set_pixels(int r, int g, int b, int brightness, int count) {
  int bri = constrain(brightness, 0, 100);
  int cnt = constrain(count, 0, 8);
  ModulinoColor color(constrain(r, 0, 255), constrain(g, 0, 255), constrain(b, 0, 255));

  leds.clear();
  for (int i = 0; i < cnt; i++) {
    leds.set(i, color, bri);
  }
  leds.show();
}

void setup() {
  // VENTUNO Q Qwiic connector is on Wire1 (I2C4)
  Wire1.begin();
  Modulino.begin(Wire1);
  leds.begin();

  // Hardware self-test: light all 8 pixels cyan for 1 second on boot
  for (int i = 0; i < 8; i++) {
    leds.set(i, ModulinoColor(0, 255, 255), 40);
  }
  leds.show();
  delay(1000);

  // Set pixel 0 to dim green to indicate "Ready for RPC"
  leds.clear();
  leds.set(0, GREEN, 20);
  leds.show();

  Bridge.begin();
  Bridge.provide_safe("set_pixels", set_pixels);
}

void loop() {
  // Arduino_RouterBridge services provide_safe() callbacks automatically in the main loop context
  delay(1);
}
