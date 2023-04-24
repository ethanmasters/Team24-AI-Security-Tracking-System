//
//  prototypeApp.swift
//  prototype
//
//  Created by Sarah Beltran on 3/15/23.
//

import SwiftUI
import Firebase
//import FirebaseFirestore
//import FirebaseAuth
      
@main
struct prototypeApp: App {
    
    @UIApplicationDelegateAdaptor(AppDelegate.self) var appDelegate
    
    var body: some Scene {
        WindowGroup {
            ContentView(selectedState: .cam1)
        }
    }
}

class AppDelegate: NSObject, UIApplicationDelegate {
    func application(_ application: UIApplication, didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey : Any]? = nil) -> Bool {
        FirebaseApp.configure()
        print("Configured Firebase!")
        
        return true
    }
}
