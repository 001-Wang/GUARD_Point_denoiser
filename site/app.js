'use strict';

// Values copied from tab:robust_miou_extended in cas-dc-template.tex.
const benchmarks = [
  {backbone:'PointNet++',method:'Vanilla',values:[.7739,.5259,.8043,.6082,.6543,.3242,.8091,.5073,.7175,.4729,.6757,.6221]},
  {backbone:'PointNet++',method:'PointCVaR',values:[.6685,.5281,.7840,.6049,.7244,.4125,.7851,.5207,.7049,.5699,.6672,.6473]},
  {backbone:'PointNet++',method:'GUARD',values:[.8318,.5601,.8176,.7014,.8111,.4409,.8206,.6131,.7670,.6396,.7220,.7095]},
  {backbone:'DGCNN',method:'Vanilla',values:[.7434,.4865,.7878,.6200,.7421,.4310,.7902,.5582,.7139,.5879,.6805,.6608]},
  {backbone:'DGCNN',method:'PointCVaR',values:[.6964,.4867,.8001,.6555,.7106,.4051,.8032,.6266,.7378,.5605,.7205,.6729]},
  {backbone:'DGCNN',method:'GUARD',values:[.7425,.5125,.8293,.7320,.7471,.5202,.8258,.6819,.7855,.6479,.7530,.7288]}
];
const metricIndex = {hdd:0,scannet:1,shape:11,ghost5:3,global5:5,local5:7};
const backboneSelect = document.querySelector('#backbone');
const benchmarkSelect = document.querySelector('#benchmark');

function renderBenchmark() {
  const rows=benchmarks.filter(row=>row.backbone===backboneSelect.value);
  const column=metricIndex[benchmarkSelect.value];
  const chart=document.querySelector('#benchmark-chart');
  chart.replaceChildren();
  rows.forEach((row,index)=>{
    const wrapper=document.createElement('div');
    wrapper.className=`bar-row ${['vanilla','cvar','guard'][index]}`;
    const name=document.createElement('span'); name.className='bar-name';name.textContent=row.method;
    const track=document.createElement('div');track.className='bar-track';track.setAttribute('aria-hidden','true');
    const fill=document.createElement('div');fill.className='bar-fill';fill.style.width=`${row.values[column]*100}%`;track.append(fill);
    const value=document.createElement('span');value.className='bar-value';value.textContent=row.values[column].toFixed(4);
    wrapper.append(name,track,value);chart.append(wrapper);
  });
  const delta=(rows[2].values[column]-rows[0].values[column])*100;
  const note=document.querySelector('#result-note');
  if(delta<0){note.textContent=`DGCNN + GUARD is ${Math.abs(delta).toFixed(2)} percentage points below the vanilla backbone on HDD. Filtering can remove informative local edge structure; see the trade-off discussion below.`;}
  else{note.textContent=`GUARD improves ${backboneSelect.value} by ${delta.toFixed(2)} percentage points over the vanilla backbone in this evaluation.`;}
}
backboneSelect.addEventListener('change',renderBenchmark);
benchmarkSelect.addEventListener('change',renderBenchmark);
renderBenchmark();

const fullTable=document.querySelector('#full-results tbody');
benchmarks.forEach((row,index)=>{
  const tr=document.createElement('tr');
  if(row.method==='GUARD')tr.classList.add('ours-row');
  if(index===3)tr.classList.add('group-start');
  [row.backbone,row.method,...row.values.map(v=>v.toFixed(4))].forEach((value,i)=>{
    const cell=document.createElement(i===0?'th':'td');
    if(i===0)cell.scope='row';cell.textContent=value;tr.append(cell);
  });fullTable.append(tr);
});

document.querySelector('#hero-sample').addEventListener('change',event=>{
  const id=event.target.value;
  [['original',0,'Original'],['cvar',2,'PointCVaR'],['guard',1,'GUARD']].forEach(([key,suffix,label])=>{
    const img=document.querySelector(`#hero-${key}`);
    img.src=`assets/visualization/HDD${id}_${suffix}.png`;
    img.alt=`${label} result for real HDD scan ${Number(id)+1}`;
  });
});

const gallerySets={hdd:[['HDD0','HDD 01'],['HDD1','HDD 02'],['HDD2','HDD 03'],['HDD3','HDD 04']],shape:[['chair0','Chair 01'],['chair1','Chair 02'],['guitar0','Guitar 01'],['guitar1','Guitar 02'],['lamp0','Lamp 01'],['lamp1','Lamp 02'],['laptop0','Laptop 01'],['laptop1','Laptop 02']]};
function renderGallery(){
  const set=gallerySets[document.querySelector('#gallery-dataset').value];
  const gallery=document.querySelector('#gallery');gallery.replaceChildren();
  gallery.style.gridTemplateColumns=`90px repeat(${set.length},minmax(145px,1fr))`;
  const corner=document.createElement('span');corner.className='gallery-head';corner.textContent='Method';gallery.append(corner);
  set.forEach(([,label])=>{const h=document.createElement('span');h.className='gallery-head';h.textContent=label;gallery.append(h);});
  [['Original',0],['PointCVaR',2],['GUARD',1]].forEach(([method,suffix])=>{
    const label=document.createElement('span');label.className=`row-label ${method==='GUARD'?'guard':''}`;label.textContent=method;gallery.append(label);
    set.forEach(([prefix,name])=>{
      const link=document.createElement('a');link.className='gallery-image';link.href=`assets/visualization/${prefix}_${suffix}.png`;link.target='_blank';link.rel='noopener';
      link.setAttribute('aria-label',`Open ${name}, ${method}, full resolution`);
      const img=document.createElement('img');img.src=link.href;img.loading='lazy';img.alt=`${name}: ${method}`;link.append(img);gallery.append(link);
    });
  });
}
document.querySelector('#gallery-dataset').addEventListener('change',renderGallery);
renderGallery();

document.querySelector('#copy-citation').addEventListener('click',async()=>{
  const code=document.querySelector('#bibtex');const status=document.querySelector('#copy-status');
  try{await navigator.clipboard.writeText(code.textContent);status.textContent='Citation copied.';}
  catch{const range=document.createRange();range.selectNodeContents(code);const selection=window.getSelection();selection.removeAllRanges();selection.addRange(range);status.textContent='Citation selected. Press Ctrl+C (or Command+C) to copy.';}
});

if('IntersectionObserver' in window){
  const navLinks=[...document.querySelectorAll('.outline a')];
  const observer=new IntersectionObserver(entries=>{entries.forEach(entry=>{if(entry.isIntersecting){navLinks.forEach(link=>{const active=link.hash===`#${entry.target.id}`;link.classList.toggle('active',active);if(active)link.setAttribute('aria-current','location');else link.removeAttribute('aria-current');});}});},{rootMargin:'-15% 0px -65% 0px'});
  document.querySelectorAll('main section[id]').forEach(section=>observer.observe(section));
}
