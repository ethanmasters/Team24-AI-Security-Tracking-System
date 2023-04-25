//
//  File.swift
//  prototype
//
//  Created by Sarah Beltran on 4/18/23.
//

import Foundation
import SwiftUI
struct WriteView: View{
    
    @StateObject
    var viewModel = WriteViewModel()
    
    
    @State
    var content: String = ""
    
    @State
    var refrencedProfile: String = ""
    
    @State
    var customName: String = ""
    
    var body: some View{
        VStack{
            TextEditor(text: $content)
                .frame(width: .infinity, height: 50)
                .padding()
            
            Button {
                viewModel.pushNewValue(value: content)
            } label: {
                Text("Push")
                    .padding()
            }
            
            Button {
                viewModel.pushObject()
            } label: {
                Text("Push Object")
                    .padding()
            }
            
            Button {
                viewModel.pushArrayObject()
            } label: {
                Text("Push Array Object")
                    .padding()
            }
            Text("PROFILE:")
            TextEditor(text: $refrencedProfile)
                .frame(width: .infinity, height: 50)
                .padding()
            
            Text("NAME:")
            TextEditor(text: $customName)
                .frame(width: .infinity, height: 50)
                .padding()
            
            Button {
                viewModel.pushName(value: Int(refrencedProfile) ?? 0, content: customName)
            } label: {
                Text("Write Custom Name")
                    .padding()
            }
        }.frame(width: .infinity,alignment: .top)
    }

}




struct Write_Preview: PreviewProvider{
    static var previews: some View{
        WriteView()
    }
}
