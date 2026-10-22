#include "core.h"
DiskBackend::DiskBackend(std::string,size_t) {}
std::string compute_disk_backup_dir_from_env() {return "";}
size_t compute_disk_chunk_bytes_from_env() {return 1;}
void DiskBackend::offload(void*p,size_t n,DiskBackupSlot&s) {
    static unsigned id=0;
    if(s.path.empty())s.path=std::to_string(++id);
    disk[s.path]=std::vector<char>((char*)p,(char*)p+n);++writes;
}
void DiskBackend::onload(void*p,size_t n,DiskBackupSlot&s) {
    assert(!readonly[p]); assert(disk.at(s.path).size()==n);
    std::memcpy(p,disk.at(s.path).data(),n);++reads;
}
void DiskBackend::release(DiskBackupSlot&s) {disk.erase(s.path);++releases;}
int main(int argc,char**argv) {
    auto&t=TorchMemorySaver::instance();
    void*base;void*adapter;void*other;
    t.malloc(&base,0,16,"weights",false,true);
    t.malloc(&adapter,0,8,"adapter",false,true);
    t.malloc(&other,0,4,"other",true,false);
    std::memset(base,7,16);std::memset(adapter,1,8);std::memset(other,3,4);
    const std::string mode=argc>1?argv[1]:"normal";
    if(mode=="empty")t.seal_disk_backup("");
    if(mode=="missing")t.seal_disk_backup("missing");
    if(mode=="cpu")t.seal_disk_backup("other");
    if(mode=="paused") {t.pause("weights");t.seal_disk_backup("weights");}
    if(mode=="device") {void*p;t.malloc(&p,1,4,"weights",false,true);t.seal_disk_backup("weights");}
    // A snapshot made before sealing is invalidated, even if it already exists.
    t.pause("weights");t.resume("weights");std::memset(base,9,16);
    assert(t.seal_disk_backup("weights")==16);
    assert(t.seal_disk_backup("weights")==16);
    assert(readonly[base] && !readonly[adapter] && !readonly[other]);
    for(int i=0;i<3;++i) {
        std::memset(adapter,20+i,8);
        t.pause("");
        assert(writes==unsigned(3+i)); // two base writes total, one adapter write/cycle
        t.resume("");
        assert(readonly[base] && !readonly[adapter]);
        for(int j=0;j<16;++j)assert(((char*)base)[j]==9);
        for(int j=0;j<8;++j)assert(((char*)adapter)[j]==20+i);
        for(int j=0;j<4;++j)assert(((char*)other)[j]==3);
    }
    assert(reads==7);
    t.free(base);assert(releases==1 && disk.size()==1);
    // Fresh allocation in the same tag cannot inherit the freed allocation's snapshot.
    t.malloc(&base,0,16,"weights",false,true);std::memset(base,42,16);
    assert(!readonly[base]);t.seal_disk_backup("weights");
    t.pause("weights");assert(writes==6);t.resume("weights");
    assert(((char*)base)[0]==42 && readonly[base]);
    t.free(base);t.free(adapter);t.free(other);assert(disk.empty());
}
