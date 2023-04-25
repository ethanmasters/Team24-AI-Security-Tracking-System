//
//  IndividualProfileView.swift
//  prototype
//
//  Created by Sarah Beltran on 3/26/23.
//

import SwiftUI
import FirebaseDatabase
import FirebaseDatabaseSwift

struct IndividualProfileView: View {
    private let ref = Database.database().reference()
    //var profile: Profile
    @State private var editNameTapped = false
    @State private var enterName = false
    @State private var istracked = true
    @State private var POI = true
    @State private var name: String = "Tag A"
    
    @State private var isProfileShowing: Bool
    @StateObject var viewModel = ReadViewModel()
    
    // Initialize the view with the data for the individual node
    init(node: ProfileClass) {
        _isProfileShowing = State(initialValue: node.isProfileShowing)
    }
    
    
    var body: some View {
        let _: () = viewModel.observeListObject()
        VStack(alignment: .center){
                HStack(alignment: .center){
                    VStack(alignment: .center){
                        Image(systemName: "person.circle")
                            .font(.system(size: 90))
                        Button("Edit Profile Pic") {
                            //Change Pic
                        }.foregroundColor(.white)
                    }.padding()
                    
                    VStack(alignment: .leading){
                        if editNameTapped {
                            TextField("Enter Name", text: $name)
                                .background(Color.black)
                                .foregroundColor(.white)
                                .padding()
                        }
                        Text("\(name)")
                            .font(.title)
                            .bold()
                            .padding(.vertical,5)
                        Button(action: {
                            self.editNameTapped.toggle()
                            self.enterName.toggle()
                        }) {
                            if enterName {
                                Text("Click to Enter")
                                    .foregroundColor(.black)
                            }else{
                                Text("Edit Name")
                                
                            }
                            
                        }
                        .padding(.horizontal)
                        .background(.white)
                        
                        Toggle(isOn: $POI) {
                            if(POI == true){
                                Text("POI: YES")
                            } else {
                                Text("POI: NO")
                            }
                        }.frame(width:125)
                    }
                }
                
                Toggle(isOn: $istracked) {
                    if(istracked == true){
                        Text("Interaction Tracking: ON")
                    } else {
                        Text("Interaction Tracking: OFF")
                    }
                }.frame(width:250)
                
                Button("View Interaction History") {
                    /*@START_MENU_TOKEN@*//*@PLACEHOLDER=Action@*/ /*@END_MENU_TOKEN@*/
                }
                .padding(.horizontal)
                .padding(.vertical,5)
                .background(.white)
                
            } //HStack
            .frame(width: 400, height: 270)
            .background(Color.cyan)
            
        
    }//VStack
            
            
}
    

    
    
    struct IndividualProfileView_Previews: PreviewProvider {
        static var previews: some View {
            
            IndividualProfileView(node: node)
        }
    }

