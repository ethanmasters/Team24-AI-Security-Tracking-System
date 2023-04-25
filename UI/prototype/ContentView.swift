//
//  ContentView.swift
//  prototype
//
//  Created by Sarah Beltran on 3/15/23.
//

import SwiftUI

protocol ButtonState: CaseIterable {
    var title: String { get }
}

extension ButtonState where Self: RawRepresentable, RawValue == String {
    var title: String {
        self.rawValue
    }
}

enum CameraState: String, ButtonState {
    case cam1 = "Cam 1"
    case cam2 = "Cam 2"
    case cam3 = "Cam 3"
}

struct StateButton<T: ButtonState>: View {
    
    let states: [T]
    @State var currentIndex = 0
    @Binding var selectedState: T
    var body: some View {
        Button {
            currentIndex = currentIndex < states.count - 1 ? currentIndex + 1: 0
            selectedState = states[currentIndex]
        } label: {
            Text(states[currentIndex].title)
                //.frame(maxWidth:100)
        }.buttonStyle(CameraIcon())
    }
}

struct ContentView: View {
    @State var isDatabaseTapped = false
    @State var selectedState: CameraState
    //@State var isProfileShowing = true
    
    var body: some View {
        let DummyProfiles = [
            Profile(name: "Andy Anderson", POI: false, first_seen: "08:08 AM", last_seen: "08:30 AM", interactiontrack: false),
            Profile(name: "Carol Danvers", POI: true, first_seen: "10:08 AM", last_seen: "10:30 AM", interactiontrack: true),
            Profile(name: "Dannie Daniels", POI: false, first_seen: "09:08 AM", last_seen: "10:30 AM", interactiontrack: false),
            Profile(name: "Freddie Benson", POI: true, first_seen: "10:08 AM", last_seen: "10:30 AM", interactiontrack: true),
            Profile(name: "Tag A", POI: false, first_seen: "09:08 AM", last_seen: "10:45 AM", interactiontrack: false)
            
        ]
        
        /*
        //Profile popup
        if isProfileShowing {
            VStack{
                HStack{
                    //Spacer()
                    Button("Close"){
                        isProfileShowing = false
                    }.buttonStyle(CloseProfileIcon())
                        //.padding(.trailing,170)
                }
                IndividualProfileView()
            }
        }*/
        
        //LiveProfile View
        NavigationSplitView {
                    ListRow2()
            } detail: {
            ZStack{
                NavigationStack{
                    ZStack{
                        //Camera Background Placeholder
                        Image("Image")
                            .resizable()
                        ZStack{
                            //FaceDatabase Button
                            NavigationLink("Face Database", value: "Profile.name")
                                .buttonStyle(FaceDatabaseIcon())
                                .padding(.top,587)
                                .padding(.trailing, 980)
                        }.navigationDestination(for: String.self){ string in
                            VStack{
                                //FaceDatabase View
                                Text("Face DataBase")
                                    .font(.largeTitle)
                                    .bold()
                                List(DummyProfiles) { profile in
                                    FaceDatabaseRow(profile: profile)
                                }
                            }
                        }
                    }
                }.navigationTitle("AISTS Home")
                
                
                
                //Camera Button
                //StateButton(states: CameraState.allCases, selectedState: $selectedState)
                CameraButtonV2()
                    .padding(.leading,940)
                    .padding(.bottom,786)
                    
                //Time Label
                Label("Time", systemImage: "clock")
                        .labelStyle(TimeLable())
                        .font(.title2)
                        .padding(.bottom,860)
                
                
            }//ZStack end
            
        }//detail end
            
    } //View end
        
            
} //Content View end
        

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView(selectedState: .cam1)
    }
}

// Custom TimeLabel
struct TimeLable: LabelStyle {
    func makeBody(configuration: Configuration) -> some View {
        let Date_time = NSDate.now.formatted(date: Date.FormatStyle.DateStyle.omitted, time: Date.FormatStyle.TimeStyle.shortened)
        Text(Date_time)
            .multilineTextAlignment(.center)
        
    }
}

//Camera Icon
struct CameraIcon: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        ZStack{
            Circle()
                .foregroundColor(Color.cyan)
                .frame(width:150, height:150)
            Image(systemName: "camera")
                .resizable()
                .frame(width:75,height:60)
                .padding(.bottom,28)
                .foregroundColor(Color.white)
            configuration.label
                .padding(.top,70)
                .font(.title2).bold()
                
                
        }
        .clipShape(Circle())
        .foregroundColor(Color.white)
        .frame(width:100,height:100)
        .scaleEffect(configuration.isPressed ? 1.2 : 1)
        .animation(.easeOut(duration: 0.2), value: configuration.isPressed)
    }
}

//Face Database Button
struct FaceDatabaseIcon: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        // Icon/Look for Database Button
        ZStack{
            Image(systemName: "person.circle.fill")
                .resizable()
                .foregroundColor(Color.cyan)
                .scaledToFill()
                .frame(width:110,height:110)
                .scaleEffect(configuration.isPressed ? 1.2 : 1)
                .animation(.easeOut(duration: 0.2), value: configuration.isPressed)
            configuration.label
                .multilineTextAlignment(.center)
                .frame(width: 110,height:110)
                .foregroundColor(Color.black)
                .bold()
            }
    }
}

//Close Profile Button Icon
struct CloseProfileIcon: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        Image(systemName: "multiply.circle.fill")
            .resizable()
            .frame(width:25,height:25)
            .foregroundColor(.gray)
    }
}

///EXPERIMENTS?/////////////

// Custom Button for Realtime Profile List
struct Collapsible<Content: View>: View {
    @State var label: () -> Text
    @State var content: () -> Content
    
    @State private var collapsed: Bool = true
    
    var body: some View {
        HStack {
            Button (
                action: { self.collapsed.toggle() },
                label: {
                    //self.label()
                    Spacer()
                    VStack{
                        Image(systemName:
                                self.collapsed ? "chevron.backward" : "chevron.forward").font(.system(size: 40))
                    }
                    .background(Color.blue.opacity(0.01))
                }
            )
            .buttonStyle(PlainButtonStyle())
            HStack {
                self.content()
            }
            .frame(minWidth:0, maxWidth: collapsed ? 0 : .none, minHeight: 0, maxHeight: .infinity)
            .transition(.slide)
        }
    }
}


//Live Profile List
//struct LiveProfileList2: ListStyle {
//    func makeBody(configuration: Configuration) -> some View {
//    }
//}
struct LiveProfileList: View {
    var body: some View {
        List {
            HStack{
                Image(systemName: "person.circle")
                VStack{
                    Text("Tag A")
                    Text("POI: ")
                    Button("PROFILE") {
                        //Text("Open Profile")
                    }
                }
            }
            HStack{
                Image(systemName: "person.circle")
                VStack{
                    Text("Tag B")
                    Text("POI: ")
                    Button("PROFILE") {
                        //Text("Open Profile")
                    }
                }
            }
            HStack{
                Image(systemName: "person.circle")
                VStack{
                    Text("Tag C")
                    Text("POI: ")
                    Button("PROFILE") {
                        //Text("Open Profile")
                    }
                }
            }
            
        }
        .frame(width: 300.0)
    }
}




/*
 HStack(){
     VStack(){
         Spacer()
         
         //Face Database Button
         NavigationStack {
             Button("Face Database") {
                 isDatabaseTapped = true
             }
             .buttonStyle(FaceDatabaseIcon())
             .padding([.leading, .bottom])
         }
         NavigationLink("", destination: FaceDatabaseView(), isActive: $isDatabaseTapped)
     }
     
     
     Spacer()
     
     
     VStack() {
         // Clock Label
         Label("Time", systemImage: "clock")
             .labelStyle(TimeLable())
             .font(.title2)
         Spacer()
     }
     .padding(.leading,400)
     
     
     HStack(){
         // Camera Button
         Button("Cam 1"){
         }
         .padding(.leading,380)
         .padding(.bottom,730)
         .buttonStyle(CameraIcon())
         
         VStack(){
             // Live Profile List Tab
             Collapsible(
                 label: ({Text("People Seen")}),
                 content: {
                     VStack{
                         List {
                             HStack{
                                 Image(systemName: "person.circle")
                                     .font(.system(size: 50))
                                 VStack{
                                     Text("Tag A")
                                         .padding(.trailing,30)
                                     Text("POI: ")
                                         .padding(.trailing,30)
                                     Button("PROFILE") {
                                         //Text("Open Profile")
                                     }
                                 }
                             }
                             HStack{
                                 Image(systemName: "person.circle")
                                     .font(.system(size: 50))
                                 VStack{
                                     Text("Tag B")
                                         .padding(.trailing,30)
                                     Text("POI: ")
                                         .padding(.trailing,30)
                                     Button("PROFILE") {
                                         //Text("Open Profile")
                                     }
                                 }
                             }
                         }
                         .frame(maxHeight: .infinity)
                     }
                 }
                 
             )
             .frame(maxHeight: .infinity)
             
         }
         .frame(maxWidth: 800, maxHeight: .infinity)
     }
     
 }//HStack End
 
 
*/
