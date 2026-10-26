#pragma rtGlobals=3

// Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
// Full license: https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
// Source-inspected export helper, not yet executed in Igor.
// Load alongside LaueGo microN/IndexingN.ipf, with USE_ENERGY_STRAIN_REFINE off.
// Use a fresh output directory and an explicit revision of the running LaueGo.

Function LauelabExportIgorStrain(pattern, constrain, FullPeakList, FullPeakIndexed, outputPrefix, sourceRevision)
    Variable pattern
    String constrain, outputPrefix, sourceRevision
    Wave FullPeakList, FullPeakIndexed

#ifdef USE_ENERGY_STRAIN_REFINE
    Abort "This exporter requires the position-only DeviatoricStrainRefine branch."
#else
    if (pattern != 0)
        Abort "Use pattern 0: the audited Igor branch initializes from recip_lattice0."
    endif
    if (strlen(constrain) != 6 || strlen(sourceRevision) == 0)
        Abort "Supply a six-digit binary mask and the running LaueGo revision."
    endif
    Variable i, freeCount=0
    for (i=0; i<6; i+=1)
        if (cmpstr(constrain[i], "0") != 0 && cmpstr(constrain[i], "1") != 0)
            Abort "Mask entries must be 0 or 1."
        endif
        freeCount += str2num(constrain[i])
    endfor
    if (freeCount == 0 || cmpstr(constrain[0,2], "111") == 0)
        Abort "At least one cell parameter must vary and at least one length must be held."
    endif

    // The source routine reads every wave named by peakListWave, even when
    // FullPeakList is supplied. Preserve every such input before refinement.
    String peakLists=StringByKey("peakListWave", note(FullPeakIndexed), "=")
    Variable nLists=ItemsInList(peakLists, ",")
    if (nLists < 1)
        Abort "FullPeakIndexed must identify its original peak-list waves."
    endif
    LauelabStrainRequireNew(outputPrefix+"metadata.txt")
    LauelabStrainRequireNew(outputPrefix+"FullPeakIndexed.csv")
    LauelabStrainRequireNew(outputPrefix+"FullPeakIndexed_notes.txt")
    String names="PeaksForStrain;epsilon;epsilonBL;epsilonXHF;epsilonSample;"
    for (i=0; i<ItemsInList(names); i+=1)
        LauelabStrainRequireNew(outputPrefix+StringFromList(i,names)+".csv")
        LauelabStrainRequireNew(outputPrefix+StringFromList(i,names)+"_notes.txt")
    endfor
    for (i=0; i<nLists; i+=1)
        Wave/Z fitted=$StringFromList(i,peakLists,",")
        if (!WaveExists(fitted))
            Abort "An original fitted-peak wave is missing."
        endif
        LauelabStrainRequireNew(outputPrefix+"FullPeakList_"+num2istr(i)+".csv")
        LauelabStrainRequireNew(outputPrefix+"FullPeakList_"+num2istr(i)+"_notes.txt")
    endfor

    LauelabStrainWaveCSV(FullPeakIndexed,outputPrefix+"FullPeakIndexed.csv")
    LauelabStrainText(note(FullPeakIndexed),outputPrefix+"FullPeakIndexed_notes.txt")
    for (i=0; i<nLists; i+=1)
        Wave fitted=$StringFromList(i,peakLists,",")
        LauelabStrainWaveCSV(fitted,outputPrefix+"FullPeakList_"+num2istr(i)+".csv")
        LauelabStrainText(GetWavesDataFolder(fitted,2)+"\r"+note(fitted),outputPrefix+"FullPeakList_"+num2istr(i)+"_notes.txt")
    endfor

    String tensorPath=DeviatoricStrainRefine(pattern,constrain,coords=0,FullPeakList=FullPeakList,FullPeakIndexed=FullPeakIndexed,printIt=1)
    if (strlen(tensorPath) == 0)
        Abort "Igor refinement did not return a tensor. Input exports are incomplete evidence."
    endif
    Wave epsilon=$tensorPath
    String resultFolder=GetWavesDataFolder(epsilon,1)
    for (i=0; i<ItemsInList(names); i+=1)
        Wave/Z result=$(resultFolder+StringFromList(i,names))
        if (!WaveExists(result))
            Abort "An expected refinement diagnostic wave is missing."
        endif
        LauelabStrainWaveCSV(result,outputPrefix+StringFromList(i,names)+".csv")
        LauelabStrainText(note(result),outputPrefix+StringFromList(i,names)+"_notes.txt")
    endfor
    String metadata
    sprintf metadata, "format=lauelab-igor-strain-evidence-v1\rpattern=%d\rfit_mask=%s\rigor_version=%.17g\rlauego_revision=%s\r", pattern,constrain,IgorVersion(),sourceRevision
    metadata += "objective=position-only uniform direction chord\r"
    metadata += "reciprocal_note_layout=column vectors, 1/nm including 2*pi\r"
    metadata += "csv_layout=zero-based row,column,layer,value; last index varies fastest\r"
    metadata += "wave_values=17 significant digits; reciprocal/lattice notes rounded by Igor source\r"
    metadata += "required_external_evidence=active geometry XML, effective reference, original image/ROI/group/depth, rematching audit\r"
    metadata += "peak_list_waves="+peakLists+"\r"
    LauelabStrainText(metadata,outputPrefix+"metadata.txt")
#endif
    return 0
End

Static Function LauelabStrainRequireNew(path)
    String path
    GetFileFolderInfo/Q/Z path
    if (V_flag == 0)
        Abort "Export destination exists: "+path
    endif
End

Static Function LauelabStrainText(contents, path)
    String contents, path
    Variable handle
    Open handle as path
    fprintf handle, "%s", contents
    Close handle
End

Static Function LauelabStrainWaveCSV(values, path)
    Wave values
    String path
    Variable handle, i, j, k, element
    Variable rows=DimSize(values,0), cols=max(DimSize(values,1),1), layers=max(DimSize(values,2),1)
    Open handle as path
    fprintf handle, "row,column,layer,value\r"
    for (i=0; i<rows; i+=1)
        for (j=0; j<cols; j+=1)
            for (k=0; k<layers; k+=1)
                if (DimSize(values,1) == 0)
                    element = values[i]
                elseif (DimSize(values,2) == 0)
                    element = values[i][j]
                else
                    element = values[i][j][k]
                endif
                fprintf handle, "%d,%d,%d,%.17g\r", i,j,k,element
            endfor
        endfor
    endfor
    Close handle
End
