//
//  Readview.swift
//  prototype
//
//  Created by Sarah Beltran on 4/18/23.
//

import Foundation
import SwiftUI
struct ReadView: View{
    @StateObject
    var viewModel = ReadViewModel()
    var body: some View{
        VStack{
            if !viewModel.listProfiles.isEmpty {
                VStack{
                    ForEach(viewModel.listProfiles) { object in
                        VStack{
                            Text(object.id)
                            Text(object.name)
                            Text(object.first_seen)
                        }.padding()
                    }
                }
            } else {
                Text("The place to displace our value here...")
                    .padding()
                    .background(Color.gray)
            }
            Button {
                viewModel.observeListObject()
            } label: {
                Text("Read")
                    .padding()
                    .background(Color.blue)
                    .foregroundColor(.white)
            }
        }
    }
    
}

struct Read_Preview: PreviewProvider{
    static var previews: some View{
        ReadView()
    }
}
