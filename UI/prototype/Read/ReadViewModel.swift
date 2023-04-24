//
//  ReadViewModel.swift
//  prototype
//
//  Created by Sarah Beltran on 4/18/23.
//

import Foundation
import FirebaseDatabase
import FirebaseDatabaseSwift


class ReadViewModel: ObservableObject{
    
    var ref = Database.database().reference()
    
    @Published
    var value: String? = nil
    
    @Published
    var object: ProfileClass? = nil
    
    @Published
    var listProfiles = [ProfileClass]()
    
    func readValue(){
        ref.child("0").observeSingleEvent(of: .value) { snapshot in
            self.value = snapshot.value as? String ?? "Load failed"
        }
    }
    
    func observeDataChange(){
        ref.child("0").observe(.value) { snapshot in
            self.value = snapshot.value as? String ?? "Load failed"
        }
    }
    
    func readObject(){
        ref.child("1")
            .observe(.value) { snapshot in
                do{
                    self.object = try snapshot.data(as: ProfileClass.self)
                }catch{
                    print("Can not convert to ProfileClass")
                }
            }
    }
    

    func observeListObject(){
        ref.observe(.value) { parentSnapshot in
            guard let children = parentSnapshot.children.allObjects as? [DataSnapshot] else {
                // fallout casr
                return
            }
            
            self.listProfiles = children.compactMap({ snapshot in
                return try? snapshot.data(as: ProfileClass.self)
            })
        }
    }
    
    
    
}
