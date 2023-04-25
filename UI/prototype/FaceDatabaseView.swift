//
//  FaceDatabaseView.swift
//  prototype
//
//  Created by Sarah Beltran on 3/22/23.
// TEST

import SwiftUI
import FirebaseDatabase
import FirebaseDatabaseSwift

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
struct FaceDatabaseView_Previews: PreviewProvider {
    static var previews: some View {
        //FaceDatabaseView()
        FaceDatabaseView2()
    }
}

func pushTracked(value: Int, content: Bool){
    let ref = Database.database().reference()
    ref.child(String(value)).child("interactiontrack").setValue(content)
}

struct FaceDatabaseView2: View{
    private let ref = Database.database().reference()
    
    @State private var istracked = true
    @State var isProfileShowing = false
    
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
                            IndividualProfileView()
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
                                Toggle(isOn: $istracked){
                                    Toggle("Interaction Track")
                                }.onChange(of: istracked) { value in
                                   // pushTracked(value: object.id, content: value)
                                }
                                .padding(.trailing,90)
                                .padding(.top,-50)
                                    
                            }
                        }//HStack
                    }
                }//ForEach
                
            }//List
        } else {
            Button{
                viewModel.observeListObject()
            } label: {
                Text("Face Database")
            }.buttonStyle(FaceDatabaseIcon())
        }
    }//body View
}

