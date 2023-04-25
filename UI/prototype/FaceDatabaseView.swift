//
//  FaceDatabaseView.swift
//  prototype
//
//  Created by Sarah Beltran on 3/22/23.
//

import SwiftUI
import FirebaseDatabase
import FirebaseDatabaseSwift

/* ORIGINAL DESIGN FOR FACEDATABASE
    --INCLUDES HARD CODED 'DummieProfiles'
 
 --This helped me get the overall design and format down before reading/writing to the database--
 
struct FaceDatabaseRow: View {
    var profile: Profile
    @State private var istracked = true
    @State var isProfileShowing = false
    
    @StateObject
    var veiwModel = ReadViewModel()
    
    var body: some View {
        //Profile popup
        if isProfileShowing {
            VStack{
                HStack{
                    Spacer()
                    Button("Close"){
                        isProfileShowing = false
                    }.buttonStyle(CloseProfileIcon())
                        .padding(.trailing,170)
                }
                IndividualProfileView()
            }
        }
        
        HStack {
            Image(systemName: "person.circle")
                .font(.system(size: 90))
            VStack(alignment: .leading) {
                Text(profile.name)
                    .bold()
                    .font(.title)
                HStack{
                    VStack(alignment: .leading) {
                        if(profile.POI == true){
                            Text("POI: YES")
                                .font(.system(size:20))
                                .fontWeight(.semibold)
                            
                        }else{
                            Text("POI: NO")
                                .font(.system(size:20))
                                .fontWeight(.semibold)
                        }
                        Button("PROFILE") {
                            isProfileShowing = true
                        }.buttonStyle(PROFILEButton())
                    }
                    VStack(alignment: .leading){
                        Text("First Seen: " + "\(profile.first_seen)")
                            .font(.system(size:20))
                            .fontWeight(.semibold)
                        Text("Last Seen: " + "\(profile.last_seen)")
                            .font(.system(size:20))
                            .fontWeight(.semibold)
                    }
                    .padding(.leading, 25)
                    .padding(.bottom, 20)
                }
            }
            VStack {
                Text("Interaction Tracking")
                    .padding(.bottom,70)
                    .padding(.leading,80)
                    .font(.title2)
                Toggle(isOn: $istracked) {}
                    .padding(.trailing,90)
                    .padding(.top,-50)
            }
        }
    }
}

struct FaceDatabaseView: View {
    let DummyProfiles = [
        Profile(name: "Andy Anderson", POI: false, first_seen: "08:08 AM", last_seen: "08:30 AM", interactiontrack: false),
        Profile(name: "Carol Danvers", POI: true, first_seen: "10:08 AM", last_seen: "10:30 AM", interactiontrack: true),
        Profile(name: "Dannie Daniels", POI: false, first_seen: "09:08 AM", last_seen: "10:30 AM", interactiontrack: false),
        Profile(name: "Freddie Benson", POI: true, first_seen: "10:08 AM", last_seen: "10:30 AM", interactiontrack: true),
        Profile(name: "Tag A", POI: false, first_seen: "09:08 AM", last_seen: "10:45 AM", interactiontrack: false)
        
    ]
    var body: some View {
        List(DummyProfiles) { profile in
            FaceDatabaseRow(profile: profile)
        }
        
    }

}
 */

struct FaceDatabaseView_Previews: PreviewProvider {
    static var previews: some View {
        //FaceDatabaseView()
        FaceDatabaseView2()
    }
}



struct FaceDatabaseView2: View{
    private let ref = Database.database().reference()
    
    @State var isProfileShowing = false
    @State var selectedProfile: ProfileClass? = nil
    
    @StateObject
    var viewModel = ReadViewModel()
    
    var body: some View {
        
        
        if !viewModel.listProfiles.isEmpty {
            List{
                ForEach(viewModel.listProfiles){ object in
                    
                    //Profile popup
                    if (object.isProfileShowing == true) {
                        VStack{
                            HStack{
                                Spacer()
                                Button("Close"){
                                    ref.child(String(object.id)).child("isProfileShowing").setValue(false)
                                }.buttonStyle(CloseProfileIcon())
                                    .padding(.trailing,170)
                            }
                            IndividualProfileView(node: object)
                        }
                    } else{
                        HStack {
                            Image(systemName: "person.circle")
                                .font(.system(size: 90))
                            VStack(alignment: .leading) {
                                Text(object.name)
                                    .bold()
                                    .font(.title)
                                HStack{
                                    VStack(alignment: .leading) {
                                        if(object.POI == true){
                                            Text("POI: YES")
                                                .font(.system(size:20))
                                                .fontWeight(.semibold)
                                            
                                        }else{
                                            Text("POI: NO")
                                                .font(.system(size:20))
                                                .fontWeight(.semibold)
                                        }
                                        Button("PROFILE") {
                                            ref.child(String(object.id))
                                                .child("isProfileShowing").setValue(true)
                                            
                                        }.buttonStyle(PROFILEButton())
                                        
                                        
                                    }
                                    VStack(alignment: .leading){
                                        Text("First Seen: " + "\(object.first_seen)")
                                            .font(.system(size:20))
                                            .fontWeight(.semibold)
                                        Text("Last Seen: " + "\(object.last_seen)")
                                            .font(.system(size:20))
                                            .fontWeight(.semibold)
                                    }
                                    .padding(.leading, 25)
                                    .padding(.bottom, 20)
                                }
                            }
                            VStack {
                                if object.interactiontrack {
                                    Text("Interaction Tracking: ON")
                                        .padding(.bottom,70)
                                        .padding(.leading,80)
                                        .font(.title2)
                                } else {
                                    Text("Interaction Tracking: OFF")
                                        .padding(.bottom,70)
                                        .padding(.leading,80)
                                        .font(.title2)
                                }
                                Toggle("", isOn: $viewModel.listProfiles[getIndex(for: object)].interactiontrack)
                                    .padding(.trailing,90)
                                    .padding(.top,-50)
                                    .onChange(of: viewModel.listProfiles[getIndex(for: object)].interactiontrack, perform: { value in
                                        //Update Firebase when toggle changes
                                     updateTrackingValue(for: object, value: value)
                                })
                                
                                    
                            }
                        }//HStack
                    }
                }//ForEach
                
            }//List
            .sheet(item: $selectedProfile, onDismiss: {
                        // Handle dismissal
                    print("IndividualProfileView dismissed") //prints in terminal
                    }, content: { profile in
                        IndividualProfileView(node: profile)
                    })
        }else {
            Button{
                viewModel.observeListObject()
            } label: {
                Text("Face Database")
            }.buttonStyle(FaceDatabaseIcon())
        }
    }//body View
        
    
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
}

