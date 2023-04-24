//
//  ProfileClass.swift
//  prototype
//
//  Created by Sarah Beltran on 4/19/23.
//

import Foundation
class ProfileClass: Encodable, Decodable, Identifiable{
    var id:String = ""
    var name: String = ""
    var first_seen: String = ""
    var last_seen: String = ""
    var POI: Bool = false
    var interactiontrack: Bool = true
    var isProfileShowing: Bool = false
    
    
}


extension Encodable{
    var toDictionary: [String: Any]?{
        guard let data = try? JSONEncoder().encode(self) else {
            return nil
        }
        
        return try? JSONSerialization.jsonObject(with: data, options: .allowFragments) as? [String: Any]
    }
}
