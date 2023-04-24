//
//  WriteViewModel.swift
//  prototype
//
//  Created by Sarah Beltran on 4/18/23.
//

import Foundation
import FirebaseDatabase
import FirebaseDatabaseSwift

class WriteViewModel: ObservableObject{
    
    private let ref = Database.database().reference()
    
    private var number = 0
    
    func pushNewValue(value: String){
        //ref.setValue(value) //Set Value to Database
        //ref.childByAutoId().setValue(value) //Set value to Random child
        ref.child("name").setValue(value) //Set value to Specific Child Name
    }
    
    func pushObject(){
        let generateObject = ProfileClass()
        generateObject.id = String(number)
        generateObject.name = "Profile \(number)"
        generateObject.first_seen = "00:00AM"
        generateObject.last_seen = "00:00AM"
        generateObject.POI = false
        generateObject.interactiontrack = true
        generateObject.isProfileShowing = false
        
        //ref.child(generateObject.id).setValue(generateObject.toDIctionary) //Write w/ Specific Object ID
        ref.childByAutoId().setValue(generateObject.toDictionary) //Write w/ Random Object ID
        
        number+=1
    }
    
    func pushName(value: Int, content: String){
        ref.child(String(value)).child("name").setValue(content)
    }
    
    
    
    func pushArrayObject(){
        var array = Array<Any>()
        
        for i in 0...4 {
            let generateObject = ProfileClass()
            generateObject.id = String(i)
            generateObject.name = "Profile \(i)"
            generateObject.first_seen = "00:00AM"
            generateObject.last_seen = "00:00AM"
            generateObject.POI = false
            generateObject.interactiontrack = true
            generateObject.isProfileShowing = false
            
//            array.append(generateObject.toDictionary)
            ref.child(String(i)).setValue(generateObject.toDictionary)
        }
        
    }
    
}

