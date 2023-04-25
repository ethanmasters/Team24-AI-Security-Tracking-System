//
//  IndividualProfileView.swift
//  prototype
//
//  Created by Sarah Beltran on 3/26/23.
//

import SwiftUI
import FirebaseDatabase
import FirebaseDatabaseSwift
import Combine

struct ParentView: View {
    @StateObject var viewModel = ReadViewModel() //Create instance of ReadViewModel
    var body: some View {
        IndividualProfileView(node: ProfileClass())
            .environmentObject(viewModel) //pass viewModel as enviornment object
    }
}
struct IndividualProfileView: View {
    @EnvironmentObject var viewModel: ReadViewModel
    //@EnvironmentObject private var readViewModel = ReadViewModel
    
    //@Environment(\.presentationMode) var presentationMode
    
    private let ref = Database.database().reference()
    var node: ProfileClass
    
    //State properties that track Profile editing mode
    @State private var isEditingName = false
    @State private var editedName = ""
    
    // Function to update the Firebase Realtime Database with the edited name
    func pushName(value: Int, content: String){
        ref.child(String(value)).child("name").setValue(content)
    }
    
    // Function to update the Firebase Database with the new toggle value for interaction tracking
    func updateTrackingValue(for node: ProfileClass, value: Bool) {
        let firebaseRef = Database.database().reference()
        let childPath = "\(getIndex(for: node))/interactiontrack"
        firebaseRef.child(childPath).setValue(value)
    }
    
    // Function to get the index of the node in the local array
    func getIndex(for node: ProfileClass) -> Int {
        return viewModel.listProfiles.firstIndex(where: { $0.id == node.id }) ?? 0
    }
    
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
                }
                //Name Display
                VStack(alignment: .leading){
                    Text("\(node.name)")
                        .font(.title)
                        .bold()
                        .padding(.vertical,5)
                    //Edit Name Button
                    Button(isEditingName ? "Enter" : "Edit Name"){
                        /*
                        //if the Edit Name button is pressed you will reassign the name of the node
                        if isEditingName {
                            editedName = node.name
                        }*/
                        isEditingName.toggle() //Stop edit mode when done
                        
                        // If isEditingName is false, reset the editedName to the original name
                            if !isEditingName {
                                    editedName = node.name
                            }
                    }
                    .padding(.horizontal)
                    .background(.white)
                    
                    //Show TextField only when editing mode is active
                    if isEditingName {
                        VStack{
                            TextField("Enter New Name", text: $editedName)
                                .padding()
                            // Save Button
                            Button("Save") {
                            // Reassign Name and Update everything in Database
                            let index = Int(node.id) ?? 0
                            pushName(value: index, content: editedName)
                                                    
                            // Exit editing mode and dismiss view
                            isEditingName = false
                            }
                            .foregroundColor(.black)
                            .background(Color.white)

                        }
                        
                    }
                    
                }//Name Display End
                    
            } //HStack
        }.padding().background(Color.cyan)
            
    } //View body End
} //Struct End
    
    
    
    struct IndividualProfileView_Previews: PreviewProvider {
        static var previews: some View {
            let viewModel = ReadViewModel()
            let profileNode = ProfileClass()
            
            return IndividualProfileView(node: profileNode)
                .environmentObject(viewModel)
            
        }
    }
    
    
    /*  VStack(alignment: .center){
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
     //Reassign Name and Update everying in Database
     node.name = editedName
     updateNameInDatabase()
     
     //Exit editing mode and dismiss view
     isEditingName = false
     
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
     
     if node.interactiontrack {
     Text("Interaction Tracking: ON")
     } else {
     Text("Interaction Tracking: OFF")
     }
     Text("\(node.id)")
     
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
     
     // Function to update the Firebase Database with the new toggle value for interaction tracking
     func updateTrackingValue(for node: ProfileClass, value: Bool) {
     let firebaseRef = Database.database().reference()
     let childPath = "\(getIndex(for: node))/interactiontrack"
     firebaseRef.child(childPath).setValue(value)
     }
     
     // Function to get the index of the node in the local array
     func getIndex(for node: ProfileClass) -> Int {
     return viewModel.listProfiles.firstIndex(where: { $0.id == node.id }) ?? 0
     }
     */
    

