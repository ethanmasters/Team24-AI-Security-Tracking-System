//
//  SwiftUITestSplitView.swift
//  prototype
//
//  Created by Sarah Beltran on 3/24/23.
//

import SwiftUI

struct SwiftUITestSplitView: View {
    var body: some View {
        NavigationSplitView {
            /*@START_MENU_TOKEN@*/Text("Sidebar")/*@END_MENU_TOKEN@*/
        } detail: {
            /*@START_MENU_TOKEN@*/Text("Detail")/*@END_MENU_TOKEN@*/
        }
    }
}

struct CameraButtonV2: View {
    @State private var selectedCams = "Cam 1"
    @State private var CameraPickerVisible = false
    let Cams = ["Cam 1", "Cam 2", "Cam 3"]
    
    var body: some View {
        VStack {
            Button(action: {
                //Open Picker
                self.CameraPickerVisible.toggle()
                
            }) {
                Text("\(selectedCams)")
            }
            .buttonStyle(CameraIcon())
            .padding()
            .background(Color.white)
            if CameraPickerVisible {
                Picker(selection: $selectedCams, label: Text(""))
                    {
                        ForEach(Cams, id: \.self) {
                                Text($0)
                        }
                    }
                    
                            
            }
        }
    }
}


struct SwiftUITestSplitView_Previews: PreviewProvider {
    static var previews: some View {
        //SwiftUITestSplitView()
        CameraButtonV2()
    }
}
