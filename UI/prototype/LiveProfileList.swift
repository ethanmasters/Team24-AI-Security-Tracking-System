//
//  LiveProfileList.swift
//  prototype
//
//  Created by Sarah Beltran on 3/21/23.
//

import SwiftUI
import FirebaseDatabase
import FirebaseDatabaseSwift

protocol IdentifiableHashable: Hashable, Identifiable { }

extension IdentifiableHashable {
    // just the id's hash value is enough, ignore any other properties
    func hash(into hasher: inout Hasher) {
        hasher.combine(id)
    }
}
struct Profile: IdentifiableHashable {
    //var ppic: String
    var name: String
    var POI: Bool
    var first_seen: String
    var last_seen: String
    var interactiontrack: Bool
    var id = UUID()

}

/*
struct ListRow: View {
    @State var isProfileShowing = false
    //var profile: Profile
    
    @StateObject
    var viewModel = ReadViewModel()
    
    var body: some View {
        HStack {
            ForEach(viewModel.listProfiles){ object in
                Image(systemName: "person.circle")
                    .font(.system(size: 50))
                VStack(alignment: .leading) {
                    Text(object.name)
                        .bold()
                        .font(.title)
                    if(object.POI == true){
                        Text("POI: YES")
                            .font(.system(size:20))
                            .fontWeight(.semibold)
                        
                    }else{
                        Text("POI: NO")
                            .font(.system(size:20))
                            .fontWeight(.semibold)
                    }
                    //Profile popup
                    if isProfileShowing {
                        VStack{
                            HStack{
                                //Spacer()
                                Button("Close"){
                                    isProfileShowing = false
                                }.buttonStyle(CloseProfileIcon())
                                    .padding(.leading,200)
                            }
                            IndividualProfileView()
                                .scaledToFit()
                                .frame(width:50)
                        }
                    }
                    Button("PROFILE") {
                        isProfileShowing = true
                    }.buttonStyle(PROFILEButton())
                }
            }
            
        }
    }
}
 
struct CustomListView: View {
    let DummyProfiles = [
        Profile(name: "Andy Anderson", POI: false, first_seen: "08:08 AM", last_seen: "08:30 AM", interactiontrack: false),
        Profile(name: "Carol Danvers", POI: true, first_seen: "10:08 AM", last_seen: "10:30 AM", interactiontrack: true),
        Profile(name: "Dannie Daniels", POI: false, first_seen: "09:08 AM", last_seen: "10:30 AM", interactiontrack: false),
        Profile(name: "Freddie Benson", POI: true, first_seen: "10:08 AM", last_seen: "10:30 AM", interactiontrack: true),
        Profile(name: "Tag A", POI: false, first_seen: "09:08 AM", last_seen: "10:45 AM", interactiontrack: false)
        
    ]
    var body: some View {
        /*List(DummyProfiles) { profile in
            ListRow(profile: profile)
        }*/
        ListRow()
        
    }
    
}
*/
    
//Collapse Button
struct CollapseIcon: ButtonStyle {
    @State private var collapsed: Bool = true
    func makeBody(configuration: Configuration) -> some View {
        Button (
            action: {self.collapsed.toggle()},
            label: {
                Image(systemName:
                        self.collapsed ? "cheveron.backward" :
                        "cheveron.forward").font(.system(size: 40))
            }
        )
        .buttonStyle(PlainButtonStyle())
    }
}

//PROFILE Button
struct PROFILEButton: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .bold()
            .padding([.leading,.trailing],10)
            .padding([.top,.bottom],4)
            .background(Color.cyan)
            .foregroundColor(Color.white)
            .clipShape(RoundedRectangle(cornerRadius: 2, style: .continuous))
            .scaleEffect(configuration.isPressed ? 1.2 : 1)
            .animation(.easeOut(duration: 0.2), value: configuration.isPressed)
            
    }
}

struct CustomListView_Previews: PreviewProvider {
    static var previews: some View {
        ListRow2()
    }
}


struct ListRow2: View {
    private let ref = Database.database().reference()
    
    @State var isProfileShowing = false
    @StateObject
    var viewModel = ReadViewModel()
    @State var selectedProfile: ProfileClass? = nil
    
    var body: some View {
       
        if !viewModel.listProfiles.isEmpty {
            List{
                ForEach(viewModel.listProfiles){ object in
                    HStack{
                        Image(systemName: "person.circle")
                            .font(.system(size: 50))
                        VStack(alignment: .leading){
                            Text(object.name)
                                .bold()
                                .font(.title)
                            
                            if (object.POI == false){
                                Text("POI: NO")
                            } else {
                                Text("POI: YES")
                            }
                            
                            Button("PROFILE"){
                                selectedProfile = object
                            }.buttonStyle(PROFILEButton())
                            
                            //Profile popup
                            if (object.isProfileShowing == true){
                                VStack{
                                    HStack{
                                        //Spacer()
                                        Button("Close"){
                                            ref.child(String(object.id)).child("isProfileShowing").setValue(false)
                                        }.buttonStyle(CloseProfileIcon())
                                            .padding(.leading,200)
                                    }
                                    IndividualProfileView(node: object)
                                } //End VStack
                            }//End if Statement
                        } //End VStack
                    }//End HStack
                }//End ForEach
            } //End List
            .sheet(item: $selectedProfile, onDismiss: {
                        // Handle dismissal
                    print("IndividualProfileView dismissed") //prints in terminal
                    }, content: { profile in
                        IndividualProfileView(node: profile)
                    })
        } else {
            Button{
                viewModel.observeListObject()
            } label: {
                Text("View Live Profile List")
            }
            
        }
    }
}
