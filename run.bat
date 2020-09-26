
REM tscon 1 /dest:console

python record.py --num-items 50000 --json-config d:\dev\pytorch-blender-dr\blender\tless\bgImgConfig.json tless
REM python record.py --num-items 20 --json-config d:\dev\pytorch-blender-dr\blender\tless\bgImgConfig.json tless

xcopy /Y .\tmp\*.btr \\gpusrv.profactor.local\data\20200924_TLess_asInNoiseBranchButRealBGImages\

xcopy /Y .\tmp\output_0.png \\gpusrv.profactor.local\data\20200924_TLess_asInNoiseBranchButRealBGImages\
xcopy /Y .\tmp\output_100.png \\gpusrv.profactor.local\data\20200924_TLess_asInNoiseBranchButRealBGImages\
xcopy /Y .\tmp\output_1000.png \\gpusrv.profactor.local\data\20200924_TLess_asInNoiseBranchButRealBGImages\
xcopy /Y .\tmp\output_10000.png \\gpusrv.profactor.local\data\20200924_TLess_asInNoiseBranchButRealBGImages\

xcopy /Y d:\dev\pytorch-blender-dr\blender\tless\bgImgConfig.json \\gpusrv.profactor.local\data\20200924_TLess_asInNoiseBranchButRealBGImages\
