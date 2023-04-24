/*
Team 24- AI tracking system 
Author - Karthik Ravikumar
Hardware subsystem for lipo fuel gauge. Below is the arduino code for interfacing the arduino with the lipo fuel gauge and OLED display.
*/

#include <U8glib.h>
#include <Adafruit_SH1106.h>
#include <SPI.h>
#include <Wire.h>
#include <LiFuelGauge.h>        
#include <Adafruit_GFX.h>

 
#define SCREEN_WIDTH 128    // OLED display width, in pixels
#define SCREEN_HEIGHT 64    // OLED display height, in pixels
#define OLED_RESET     -1    // Reset pin # (or -1 if sharing Arduino reset pin)
#define SCREEN_ADDRESS 0x3C // 0x3D for 128x64, 0x3C for 128x32
Adafruit_SH1106 display(OLED_RESET);
 
 
void lowPower();
 
LiFuelGauge gauge(MAX17043, 0, lowPower); //initalizes fuel gauge
 
volatile boolean alert = false;
 
 
void setup()
{
  Serial.begin(9600); // Initializes serial port
  /*
  if (!display.begin(SH1106_SWITCHCAPVCC,0x3C))
  {
    Serial.println(F("SH1106 allocation failed"));
    for (;;); // Don't proceed, loop forever
  }
  
  */
  pinMode(20, INPUT); //sda

  //initalizes display
  display.begin(SH1106_SWITCHCAPVCC, SCREEN_ADDRESS);
  display.display();
  delay(200);
  display.clearDisplay();
 
  // Waits for serial port to connect
  while ( !Serial ) ;
 
  gauge.reset();  // Resets MAX17043
  delay(200);  // Waits for the initial measurements to be made
  
  // Sets the Alert Threshold to 10% of full capacity
  gauge.setAlertThreshold(10);
  Serial.println(String("Alert Threshold is set to ") +
                 gauge.getAlertThreshold() + '%');

}
 
void loop()
{
  Serial.print("SOC: ");
  Serial.print(gauge.getSOC());  // Gets the battery's state of charge
  Serial.print("%, VCELL: ");
  Serial.print(gauge.getVoltage());  // Gets the battery voltage
  Serial.println('V');
 
  display.setCursor(0, 10); //oled display
  display.setTextSize(2);
  display.setTextColor(WHITE);
  display.print("SOC:");
  display.print(gauge.getSOC());
  display.print("%");
 
  display.setCursor(0, 40); //oled display
  display.setTextSize(2);
  display.setTextColor(WHITE);
  display.print("VOL:");
  display.print(gauge.getVoltage());
  display.print("V");
 
  display.display();
  display.clearDisplay();

  // triggers when battery level reaches below 10%
  if ( alert )
  {
    Serial.println("Beware, Low Power!");
    Serial.println("Finalizing operations...");
    gauge.clearAlertInterrupt();  // Resets the ALRT pin
    alert = false;
    Serial.println("Storing data...");
    Serial.println("Sending notification...");
    Serial.println("System operations are halted...");
    gauge.sleep();  // Forces the MAX17043 into sleep mode
    while ( true ) ;
  }
  delay(2000);
  

  //int sda = digitalRead(20);
  //delay(100);
}
 
void lowPower()
{
  alert = true;
}