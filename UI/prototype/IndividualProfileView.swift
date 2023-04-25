//
//  IndividualProfileView.swift
//  prototype
//
//  Created by Sarah Beltran on 3/26/23.
//

import SwiftUI
import FirebaseDatabase
import FirebaseDatabaseSwift

struct ParentView: View {
    @StateObject var viewModel = ReadViewModel() //Create instance of ReadViewModel
    var body: some View {
        IndividualProfileView(node: ProfileClass())
            .environmentObject(viewModel) //pass viewModel as enviornment object
    }
}
struct IndividualProfileView: View {
    @EnvironmentObject var viewModel: ReadViewModel
    @StateObject private var readViewModel = ReadViewModel()
    
    @Environment(\.presentationMode) var presentationMode
    
    private let ref = Database.database().reference()
    var node: ProfileClass
    
    //State properties that track Profile editing mode
    @State private var isEditingName = false
    @State private var editedName = ""
    
    @State private var editNameTapped = false
    @State private var enterName = false
    @State private var istracked = true
    @State private var POI = true
    @State private var name: String = "Tag A"
    
    
    
    
    var body: some View {
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
                        Text("\(node.name)")
                            .font(.title)
                            .bold()
                            .padding(.vertical,5)
                        //Edit Name Button
                        Button(isEditingName ? "Cancel" : "Edit Name"){
                            //if the Edit Name button is pressed you will reassign the name of the node
                            if isEditingName {
                                editedName = node.name
                            }
                            isEditingName.toggle() //Stop edit mode when done
                        }
                        .padding(.horizontal)
                        .background(.white)
                        
                        //Show TextField only when editing mode is active
                        if isEditingName {
                            VStack{
                                TextField("Enter New Name", text: $editedName)
                                    .padding()
                                
                                Button("Save"){
                                    //Update everying edited
                                    node.name = editedName
                                    ref.child(String(node.id)).child("name").setValue(node.name)
                                    
                                    //Exit editing mode and dismiss view
                                    isEditingName = false
                                    presentationMode.wrappedValue.dismiss()
                                }
                            }
                        }
                        
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
            let viewModel = ReadViewModel()
            let profileNode = ProfileClass()
            
            return IndividualProfileView(node: profileNode)
                .environmentObject(viewModel)
            
        }
    }

