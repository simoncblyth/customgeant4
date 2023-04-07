#include <cassert>
#include "C4Track.h"

int main(int argc, char** argv)
{
    G4Track* track = C4Track::MakePhoton() ;  
    assert(track); 

    std::cout << "0: " << C4Track::Desc(track) << std::endl; 

    C4Track::SetLabel(track, 1,2,3, 10,20,30,40 ); 

    std::cout << "1: " << C4Track::Desc(track) << std::endl; 

    int gs ; 
    int ix ; 
    int id ; 
    int gen ; 
    int eph ; 
    int ext ; 
    int flg ; 

    C4Track::GetLabel(track, gs, ix, id, gen, eph, ext, flg ); 

    std::cout << "2: " << C4Track::Desc(track) << std::endl; 


    assert( gs == 1 );  
    assert( ix == 2 );  
    assert( id == 3 );  
    assert( gen == 10 );  
    assert( eph == 20 );  
    assert( ext == 30 );  
    assert( flg == 40 );  

    int flg1 = C4Track::GetLabelFlag( track ); 
    assert( flg1 == flg ); 

    C4Track::SetLabelFlag( track, 255 ); 

    std::cout << "3: " << C4Track::Desc(track) << std::endl; 


    int flg2 = C4Track::GetLabelFlag( track ); 
    assert( flg2 == 255 ); 
    std::cout << "4: " << C4Track::Desc(track) << std::endl; 


    return 0 ; 
}
