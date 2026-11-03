# CareLoop-Edge: demonstration video script (≈ 8 minutes)

**Story.** Tanmay, a patient at a rural health centre, brings his brain MRI on a pen drive and uploads it at the
CareLoop kiosk. A radiologist reports it from a laptop in the city. Tanmay reads his report and referral letter on his
phone and books a consultation. The care coordinator makes sure he actually reaches the neurosurgeon. That last step is
the loop in the name CareLoop: no patient is lost between "scan done" and "treatment started".

**Cast and devices**

| Role | Account | Device | Address |
|---|---|---|---|
| Patient, **Tanmay** | created live at the kiosk | Kiosk: Raspberry Pi 4 + 7" touch screen, pen drive | opens by itself (`https://localhost`) |
| Patient, **Tanmay**, later | same | his phone (Gmail + browser, **mobile data**) | `https://careloop-pi4.taild55bc9.ts.net` |
| Radiologist, **Dr. Madhur Saxena** | `doctor@clinic.org` | Laptop, browser 1 | same address → **Clinician** |
| Care coordinator, **Rekha** | `careloop@clinic.org` (admin) | Laptop, browser 2 (incognito) | same address → **Clinician** |

Also on the Pi:
- Dr. Sandeep Saini (`medicine@clinic.org`);
- Dr. Kusum Lata, general physician (`kusum@lnmiit.ac.in`);
- patient Prashant (`21pec002@lnmiit.ac.in`);
- the six demo patients listed in Part A.

Narration is in *italics*. **[SCREEN]** says what is filmed; **[DO]** says what to tap or type; **[CUT]** marks a wait
to remove in editing.

---

## Part A: Preparation

### Already done on the Pi
- **Public address:** `https://careloop-pi4.taild55bc9.ts.net` (Tailscale Funnel). It is also the kiosk QR code and
  the address in every e-mail link.
- **E-mail:** sends from `careloop.clinic@gmail.com`, with the clinic's name as the sender.
- **AI model:** `mri 0.3.0`, signed and verified on the device. A whole study takes about 50 s on the Pi 4.
- **Dr. Saxena's consultation hours:** Mon–Fri 09:00–13:00, 20-minute slots.
- **Pen drive in the Pi:** `CareLoop_Demo/MRI_Brain_Patient1` (155-file DICOM folder, a meningioma study).
- **Demo history:** six patients registered by the coordinator over the past weeks (phone only, no e-mail), plus
  Prashant. All studies are real test MRIs analysed by the AI.

  | Patient | Study (AI result) | Shows in the video |
  |---|---|---|
  | **Kavita Sharma**, Sikar, 88 km | Meningioma, signed by Dr. Saxena | Referral **Sent** 3 weeks ago and still open; calls *No answer*, then *Wrong number*. **High risk** |
  | **Sunita Devi**, Tonk, 64 km | Glioma, signed by Dr. Saini | **Repeat MRI 12 days overdue**, two unanswered calls |
  | **Mohammed Irfan**, Jaipur | No tumour | Consultation overdue, **not contacted yet**. You log a call on camera |
  | **Anjali Gupta**, Jaipur | Meningioma | Referral Sent → Appointment made → **Attended**: a **closed loop** |
  | **Ramesh Yadav**, Dausa | No tumour | Not reached for 60+ days: **Lost to follow-up** |
  | **Geeta Meena**, Alwar, 120 km | Outside report only | Coordinator's proposal **awaiting a doctor's approval** |
  | **Prashant** | Glioma, **awaiting review** | Registered by the clinic, invited by e-mail; his study waits on the worklist |

### To do before recording (about 15 min)
1. **Clinic details:** as admin, open **Settings** and fill in:
   - **Department name** (e.g. *CareLoop Rural Health Centre, Jaipur*);
   - **Address**;
   - **Telephone**.

   These appear on reports, letters, e-mails and the sign-in page.
2. **Doctor signatures:** as each doctor, open **Account → Report signature**:
   - Dr. Saxena: *MD (Radiodiagnosis), Consultant Radiologist* and his registration number;
   - Dr. Kusum Lata: *MBBS, General Physician*.
3. **Prashant:** open the invitation e-mail and tap **Set your password**.
4. **Tanmay's phone:**
   - Gmail signed in to the address he will register with;
   - **Wi-Fi off** (mobile data);
   - notifications on;
   - CareLoop mail marked **Not spam**, or a Gmail filter for `careloop.clinic@gmail.com` with "Never send to Spam".
5. **Laptop:** browser 1 signed in as Dr. Saxena; browser 2 (incognito) ready for the coordinator.
6. **Recording:**
   - laptop: OBS or Xbox Game Bar (Win+Alt+R);
   - phone: screen recorder;
   - kiosk: a second phone on a stand.
7. **When:** a **weekday afternoon**. Tanmay books **tomorrow 09:00**, which is within 24 hours, so the reminder
   e-mail arrives during the shoot.
8. **The kiosk is on the home screen** (3D brain). If not, tap the home button.

---

## Part B: The script

### Scene 1: The problem (0:00 – 0:35)

**[SCREEN]** The kiosk on a clinic desk, the 3D brain turning slowly. A slow zoom in.

*"In rural India a patient can get an MRI at a district centre, but often there's no radiologist to read it. Reports
take days, results travel on paper, and many patients are never seen again after the scan. The problem isn't only
the diagnosis. It's the loop from scan to treatment, and that loop breaks."*

*"CareLoop-Edge closes it. The brain-MRI AI runs here, on this Raspberry Pi, inside the clinic, and the scans never
leave it. A radiologist reviews and signs every report remotely. Then the system follows each patient until they
reach the right specialist."*

### Scene 2: The kiosk (0:35 – 1:05)

**[SCREEN]** Close-up of the device home: model version, status, address.

*"The kiosk shows only the state of the device, never patient data. The AI model, version 0.3.0, came signed and
encrypted from our model registry, and the device verified it before use. It reads a whole MRI study, about 36 slices
in three planes. On held-out patients it reached a tumour-type F1 score of 0.98, with a 5% false-alarm rate on healthy
scans. But it never speaks to the patient: every result goes to a radiologist first."*

**[DO]** Tap the small **QR code**. It opens full-screen; scan it with a phone, then tap to close.

*"The same portal is published securely on the internet, so the doctor can work from anywhere and the patient can
follow up from home. One scan of this code opens it."*

### Scene 3: Tanmay registers and uploads (1:05 – 2:20)

**[DO]**
1. Tap **Patient**, then **Create an account**.
2. Enter **Tanmay**, his e-mail, date of birth, sex, mobile number and a password, using the on-screen keyboard.
3. Tick consent and tap **Create account**.

*"Tanmay registers at the kiosk. New patients can sign up and upload scans only here, at the clinic, not from home.
That keeps intake under the clinic's eye and stops anyone uploading in someone else's name."*

**[DO]**
4. Tap **Upload a new scan**, then **Choose a folder (USB drive or CD)**.
5. Tap the **SanDisk** drive, then `CareLoop_Demo`, then `MRI_Brain_Patient1`.
6. In **Note for the radiologist** type *Headache for 3 months*.
7. Tick consent and tap **Submit for review**.

*"MRI centres hand patients a CD or pen drive full of DICOM files. Tanmay just picks the folder. All 155 images are
checked: one patient, one series, nothing mixed."*

**[SCREEN]** *"Uploading and analysing the study…"* **[CUT]** ~50 s.

*"The AI analyses the study right here on the Pi and puts it on the radiologist's worklist. Tanmay only sees that his
scan was received and is waiting for a radiologist. He never sees an unreviewed AI result."*

**[DO]** Show **My reports** (status *Awaiting doctor review*), then tap the home button.

*"The kiosk is shared, so every session ends by itself after a few idle minutes, and when staff use it, patient names
are shown only as initials."*

### Scene 4: Confirming the e-mail (2:20 – 2:40)

**[SCREEN]** Phone: the notification **"Please confirm your e-mail address"**.

**[DO]** Open it and tap **Confirm my e-mail address**. The page shows **E-mail address confirmed**.

*"Before any message about his health is sent, Tanmay confirms that the address really is his."*

### Scene 5: The radiologist reports remotely (2:40 – 4:25)

**[SCREEN]** Laptop, browser 1, signed in as **Dr. Madhur Saxena**: the **Reading worklist**.

*"Dr. Saxena works from the city. He opens the same address over a secure connection. His worklist is triaged:
urgent and flagged studies first, and how long each has waited. Tanmay's study has just arrived; Prashant's, from
earlier today, is waiting too."*

**[DO]** Open **Tanmay's** study.

**[SCREEN]**
1. Scroll the slice viewer through the planes.
2. Show the heat-map and lesion outline on the three key slices.
3. Show the likelihood bar along the slices and the AI finding with its confidence.

*"He sees the whole study. The bar shows the tumour likelihood on every slice. On the three most important slices
the system shows where the model looked and outlines the suspected lesion. The AI suggests meningioma, and would flag
anything unusual, such as an image unlike what it was trained on."*

**[DO]** Open the **Report** tab:
1. **Clinical indication**: already filled from Tanmay's note.
2. **Technique**: already written from the study received (DICOM series, spacing, three planes).
3. **Comparison**: *None available*.
4. Tap **Lesion** and complete the findings (location, size in mm, oedema, mass effect).
5. **Impression**: *1. Right parafalcine extra-axial mass, most consistent with meningioma.*
6. **AI suggestion**: tap **Agree**.

*"The report follows the standard radiology structure: indication, technique, comparison, findings, impression,
recommendation. Templates make it fast and consistent. Recording whether he agreed with the AI builds a running
measure of how the AI performs in real practice."*

**[DO]** Under **Follow-up**:
1. **Next step: Referral**; **Due within: 2 weeks**.
2. **Refer to**: *SMS Medical College Hospital, Jaipur*; **Specialty**: *Neurosurgery*; **Urgency**: *Within 2 weeks*.
3. **Recommendation**: *Neurosurgical consultation.*
4. Enter the password and tap **Sign and release report**.

*"This is where CareLoop differs from an ordinary reporting tool. He doesn't just write 'refer to neurosurgery' and
hope. The referral becomes a tracked step with a due date, and one electronic signature signs both the report and a
formal referral letter."*

### Scene 6: Tanmay on his phone (4:25 – 5:35)

**[SCREEN]** Phone: the notification **"Your MRI report is available"**. Open it and pause.

*"Tanmay gets an e-mail. Notice what it does not contain: no diagnosis, no findings, because e-mail isn't secure
enough for that. It says the report is ready and who reported it, and what to do if his symptoms get worse."*

**[DO]** Tap **View your report** and sign in.

**[SCREEN]** Scroll slowly:
1. the letterhead;
2. his details and the accession number;
3. the sections and the impression;
4. the signature with qualification and registration number.

*"This is the signed report, the same document the system archives as a tamper-evident copy. He can print it or save
it as a PDF."*

**[SCREEN]** **"Your referral letter is ready"** e-mail, then the **Referral letter**.

*"His referral letter is a formal doctor-to-doctor letter for the neurosurgeon, with the reason, an imaging summary
and the report enclosed. He simply carries it to the hospital."*

**[DO]** Back on the report page, tap **Book a consultation**, choose **tomorrow, 09:00**, and confirm. The page shows
**Your appointment is confirmed**, and the **"Appointment confirmed"** e-mail arrives with a calendar invite.
**[CUT]** About 5 minutes later, **"Reminder: your appointment tomorrow"** arrives, with **I'll be there** / **I can't
make it** buttons.

*"He also books a consultation to talk the report through with his radiologist. Nobody has to remember to call him:
reminders go out by themselves, 24 hours and 2 hours before, and he can confirm or cancel with one tap."*

### Scene 7: The care coordinator closes the loop (5:35 – 7:05)

**[SCREEN]** Laptop, browser 2, signed in as the coordinator: **Follow-up**.

*"Now the part most systems don't have. Rekha, the clinic's care coordinator, works the follow-up list. Every patient
with an open next step is listed, most urgent first. The risk is purely operational: days overdue, unanswered calls,
distance to the hospital. It never uses the diagnosis."*

**[SCREEN]** Point at:
- **Kavita Sharma**: *High risk*, referral sent 3 weeks ago and still not completed, *wrong number*, 88 km away;
- **Sunita Devi**: repeat MRI 12 days overdue, two unanswered calls.

**[DO]**
1. On **Mohammed Irfan** tap **Log contact**, choose **Phone call** → **Reached · will come**, and save.
2. On **Tanmay** tap **Remind**. **[CUT]** His phone shows **"Your next step: referral"**.

*"Many rural patients have no e-mail at all. Rekha calls them, and every attempt is logged. For patients with e-mail,
one tap sends a reminder."*

**[DO]** Browser 1 (Dr. Saxena): **Approvals** shows **Geeta Meena**'s consultation proposed by the coordinator.
Tap **Approve**.

*"The coordinator proposes; a doctor approves. Patients are only contacted about steps a doctor has agreed to."*

**[DO]** Browser 2: **Referrals**. Tanmay's referral to Neurosurgery is **Signed**.
1. Set it to **Sent**, then **Update**.
2. Set it to **Appointment made**, then **Update**.
3. *(caption: "two weeks later")* Set it to **Attended**, then **Update**.

*"She tracks the referral step by step: letter sent, appointment made, attended. When Tanmay is seen at the hospital,
his follow-up plan closes automatically. The loop is closed. Anjali, from last month, shows the same: attended, plan
complete."*

**[SCREEN]** **Analytics**, period **12 months**:
- loop closure;
- the funnel: reports signed → follow-up planned → patient reached → loop closed;
- **lost to follow-up** (Ramesh);
- distance bands and districts;
- reading times and AI agreement.

*"For the department, this is the number that matters: how many patients actually reached their next step after the
scan, and who was lost, and where, by distance and district. Next to it: how fast studies are read, and how often the
radiologists agreed with the AI."*

### Scene 8: Trust and security (7:05 – 7:35)

**[SCREEN]** Quick cuts:
1. **Audit log**: *Tamper-evident log verified*;
2. **System**;
3. back to the kiosk.

*"Every view, signature and change is written to a hash-chained audit log that shows any tampering. Scans and reports
are encrypted on the device, signed reports can only be amended, never edited, and the AI model only runs after its
signature is verified."*

### Scene 9: Close (7:35 – 7:55)

**[SCREEN]** The kiosk with the 3D brain, then the logo.

*"CareLoop-Edge: AI at the edge to read the scan, a radiologist to sign it, and a care loop that makes sure the patient
gets there. Built for the clinics that need it most."*

---

## Part C: Feature checklist

| CareLoop feature | Scene |
|---|---|
| On-device whole-study AI; signed, encrypted, verified model; device page | 1, 2 |
| QR code (tap to enlarge), access from anywhere | 2, 5, 6 |
| Kiosk-only sign-up, USB / DICOM folder upload, initials and auto sign-out on the kiosk | 3 |
| E-mail confirmation before any health message | 4 |
| Triaged worklist, slice viewer, heat-map and lesion outline on key slices, AI flags | 5 |
| Structured report, templates, AI agreement, electronic signature | 5 |
| Follow-up plan and signed referral letter from one signature | 5, 6 |
| E-mails with no clinical content; report and letter in the portal | 6 |
| Booking, calendar invite, automatic appointment reminder | 6 |
| Clinic-registered patient with an e-mail invitation (Prashant) | 5 |
| Follow-up list, operational risk, contact log, reminder e-mail | 7 |
| Coordinator proposes → doctor approves (Geeta) | 7 |
| Referral tracked to *Attended* → loop closed (Tanmay; Anjali) | 7 |
| Lost to follow-up (Ramesh), loop closure, distance and district, AI agreement | 7 |
| Hash-chained audit log, encryption, immutable signed reports | 8 |

## Part D: Editing and retakes
- **Cut every wait:** the ~50 s analysis, e-mail delivery, mobile page loads.
- **Role captions:** **Patient · Kiosk**, **Radiologist · Laptop**, **Patient · Phone**, **Care coordinator · Laptop**.
- **Blur** e-mail addresses in the Gmail header and any phone numbers.
- **Retake:**
  - register Tanmay with a new Gmail alias (e.g. `yourname+t2@gmail.com`);
  - or ask Claude to reset the Pi back to this prepared state (a backup of it is kept).
- **If an e-mail is slow:** the Pi retries every 5 minutes. Check Spam first.
