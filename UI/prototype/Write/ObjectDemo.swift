//
//  ObjectDemo.swift
//  prototype
//
//  Created by Sarah Beltran on 4/18/23.
//

import Foundation
class ObjectDemo: Encodable, Decodable, Identifiable{
    var id:String = ""
    var value: String = ""
    
}


extension Encodable{
    var toDIctionary: [String: Any]?{
        guard let data = try? JSONEncoder().encode(self) else {
            return nil
        }
        
        return try? JSONSerialization.jsonObject(with: data, options: .allowFragments) as? [String: Any]
    }
}
