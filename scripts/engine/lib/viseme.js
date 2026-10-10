/* Audio-aligned phonemes -> stable mouth shapes. One texture per frame.
 * Geometric easing controls jaw opening and lip width; no RMS cycling,
 * body motion, or two-mouth crossfade. Character face is supplied artwork.
 */
(function(root){
  const shapes={REST:[38,4],MBP:[36,3],FV:[34,7],A:[38,17],
                E:[36,11],I:[40,8],O:[24,19],U:[18,15]};
  function compile(data){
    if(data.phoneme_timing!=='acoustic_phoneme_ctc_forced_alignment')
      throw Error('VI: requires acoustic phoneme alignment; heuristic timing rejected');
    const result=[];let end=0;
    for(const cue of data.cues){
      if(!shapes[cue.viseme]||cue.start<end-1e-4||cue.end<=cue.start)
        throw Error('VI: invalid phoneme cue');
      if(cue.start>end+.08)result.push({start:end,end:cue.start,state:'REST'});
      const prev=result.at(-1);
      if(prev&&prev.state===cue.viseme&&cue.start-prev.end<.04)prev.end=cue.end;
      else result.push({start:cue.start,end:cue.end,state:cue.viseme});
      end=cue.end;
    }
    result.push({start:end,end:Math.max(end,data.duration),state:'REST'});
    return result;
  }
  function sample(track,t){
    const i=track.findIndex(c=>t>=c.start&&t<c.end);
    if(i<0)return {state:'REST',width:38,height:4};
    const current=track[i],previous=track[Math.max(0,i-1)];
    const [pw,ph]=shapes[previous.state],[w,h]=shapes[current.state];
    const dt=Math.min(.055,(current.end-current.start)*.65);
    let p=Math.max(0,Math.min(1,(t-current.start)/dt));p=p*p*(3-2*p);
    return {state:p<.5?previous.state:current.state,
      width:pw+(w-pw)*p,height:ph+(h-ph)*p};
  }
  const api={compile,sample,shapes};
  if(typeof module!=='undefined'&&module.exports)module.exports=api;
  else root.VI=api;
})(typeof window==='undefined'?globalThis:window);
